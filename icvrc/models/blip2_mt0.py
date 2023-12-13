from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGeneration,
    Blip2VisionModel,
    Blip2Config,
    Blip2QFormerModel,
    Blip2ForConditionalGenerationModelOutput
)
from typing import Optional, Tuple, Union
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSeq2SeqLM
import torch


class Blip2ForT5VQA(Blip2ForConditionalGeneration):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.config.use_decoder_only_language_model = False

        # Initialize weights and apply final processing
        self.post_init()


class Blip2ForT5VQAV2(Blip2ForT5VQA):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        imgtxt_input_ids: Optional[torch.LongTensor] = None,
        imgtxt_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if imgtxt_input_ids is not None:
            imgtext_embeds = self.language_model.get_input_embeddings()(imgtxt_input_ids)
            inputs_embeds = torch.cat([imgtext_embeds.to(inputs_embeds.device), inputs_embeds], dim=1)
        if imgtxt_attention_mask is not None:
            attention_mask = torch.cat([imgtxt_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        imgtxt_input_ids: Optional[torch.LongTensor] = None,
        imgtxt_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if imgtxt_input_ids is not None:
            imgtext_embeds = self.language_model.get_input_embeddings()(imgtxt_input_ids)
            inputs_embeds = torch.cat([imgtext_embeds.to(inputs_embeds.device), inputs_embeds], dim=1)
        if imgtxt_attention_mask is not None:
            attention_mask = torch.cat([imgtxt_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs


class Blip2ForT5VQAV3(Blip2ForT5VQA):
    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.qformer_projection = nn.Linear(config.text_config.hidden_size, config.qformer_config.hidden_size)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.config.use_decoder_only_language_model = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        imgtxt_input_ids: Optional[torch.LongTensor] = None,
        imgtxt_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        with torch.no_grad():
            query_to_qformer_embeds = self.language_model.get_input_embeddings()(input_ids)
            query_to_qformer_embeds *= attention_mask[:, :, None].to(query_to_qformer_embeds.dtype)

        query_to_qformer_embeds = self.qformer_projection(query_to_qformer_embeds)
        query_tokens = torch.cat([query_to_qformer_embeds.to(query_tokens.device), query_tokens], dim=1)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        query_output = query_output[:, input_ids.size()[1]:, :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_embeds = self.language_projection(query_output)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_embeds, inputs_embeds.to(language_model_embeds.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        language_model_attention_mask = torch.ones(
            query_output.size()[:-1], dtype=torch.long, device=query_tokens.device
        )
        attention_mask = torch.cat([
            language_model_attention_mask,
            attention_mask.to(language_model_attention_mask.device)
        ], dim=1)

        if imgtxt_input_ids is not None:
            imgtxt_embeds = self.language_model.get_input_embeddings()(imgtxt_input_ids)
            inputs_embeds = torch.cat([imgtxt_embeds.to(inputs_embeds.device), inputs_embeds], dim=1)
        if imgtxt_attention_mask is not None:
            attention_mask = torch.cat([imgtxt_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        imgtxt_input_ids: Optional[torch.LongTensor] = None,
        imgtxt_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        inputs_embeds = self.get_input_embeddings()(input_ids)
        query_to_qformer_embeds = inputs_embeds * attention_mask[:, :, None].to(inputs_embeds.dtype)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_to_qformer_embeds = self.qformer_projection(query_to_qformer_embeds)
        query_tokens = torch.cat([query_to_qformer_embeds.to(query_tokens.device), query_tokens], dim=1)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state
        query_output = query_output[:, input_ids.size()[1]:, :]

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if imgtxt_input_ids is not None:
            imgtxt_embeds = self.language_model.get_input_embeddings()(imgtxt_input_ids)
            inputs_embeds = torch.cat([imgtxt_embeds.to(inputs_embeds.device), inputs_embeds], dim=1)
        if imgtxt_attention_mask is not None:
            attention_mask = torch.cat([imgtxt_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
