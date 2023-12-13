from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging
import inspect
import time
import os


logger = logging.getLogger(__name__)


def ddp_setup():
    if not torch.distributed.is_initialized():
        init_process_group(backend="nccl")

    backend = torch.distributed.get_backend()
    logger.warn(f"DISTRIBUTED COMMUNICATION: {backend}")


def format_time(delta_time):
    h = int(delta_time // 3600)
    delta_time = delta_time % 3600
    m = int(delta_time // 60)
    s = round(delta_time % 60, 2)
    return f"{h}h {m}m {s}s"


class TorchTrainer():
    def __init__(
        self,
        model,
        training_args,
        train_dataset,
        eval_dataset=None,
    ):
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)

        self.training_args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_batch_per_epochs = len(train_dataset)

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.num_device = torch.cuda.device_count()
        self._model_input_args = inspect.getfullargspec(model.forward).args[1:]

        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        # dataset
        self.pt_train_dataset = self.get_pt_dataset(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size
        )
        if self.eval_dataset is not None:
            self.pt_eval_dataset = self.get_pt_dataset(
                self.eval_dataset,
                batch_size=self.training_args.per_device_train_batch_size
            )
        else:
            self.pt_eval_dataset = None

    def train(self) -> None:
        ddp_setup()

        if self.gpu_id == 0:
            # Training and validation
            logger.warn("***** Running training *****")
            logger.warn(f"  Num examples = {len(self.train_dataset)}")
            logger.warn(f"  Num Epochs = {self.training_args.num_train_epochs}")
            logger.warn(f"  Learning Rate = {self.training_args.learning_rate}")
            logger.warn(
                f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size}"
            )
            logger.warn(
                f"  Total train batch size = {self.training_args.per_device_train_batch_size * self.num_device}"
            )
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.training_args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.95)
        self.model.train()  # turn on train mode

        self.best_eval_loss = 10000.0
        total_loss = 0.0
        start_time = time.time()
        max_step = len(self.pt_train_dataset)
        for epochi in range(int(self.training_args.num_train_epochs)):
            for step, inputs in enumerate(self.pt_train_dataset):
                inputs = self.preprocess_inputs(inputs)
                optimizer.zero_grad()
                loss = self.model.forward(**inputs)[0]
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                if step % self.training_args.logging_steps == 0 and step > 0:
                    if self.gpu_id == 0:
                        lr = scheduler.get_last_lr()[0]
                        ms_per_batch = (time.time() - start_time) * 1000 / self.training_args.logging_steps
                        es_time = ((time.time() - start_time) / self.training_args.logging_steps) * (max_step - step)
                        es_time = format_time(es_time)
                        cur_loss = total_loss / self.training_args.logging_steps
                        logger.warn(
                            f'Epoch: {epochi+1}| step: {step}/{max_step} |Estimate time: {es_time} | lr {lr} | ms/batch {ms_per_batch} | loss {cur_loss}')
                        total_loss = 0.0
                        start_time = time.time()
                        torch.distributed.barrier()
                    else:
                        torch.distributed.barrier()

            if self.gpu_id == 0:
                self.hf_save_pretrained_model()
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()

            if self.pt_eval_dataset is not None:
                self.run_evaluate(epochi + 1)

        destroy_process_group()

    def run_evaluate(self, epochi):
        with torch.no_grad():
            eval_loss = torch.tensor(0.0, dtype=torch.float32).to(self.gpu_id)
            for eval_step, inputs in enumerate(self.pt_eval_dataset):
                inputs = self.preprocess_inputs(inputs)
                eval_loss += self.model.forward(**inputs)[0].to(torch.float32)
            eval_loss /= (eval_step + 1)
        torch.distributed.reduce(eval_loss, dst=0, op=torch.distributed.ReduceOp.AVG)
        if self.gpu_id == 0:
            if eval_loss < self.best_eval_loss:
                self.hf_save_best_model()
                self.best_eval_loss = eval_loss
                logger.warn(f'    - Epoch: {epochi} - Best Eval Loss: {eval_loss}')
            else:
                logger.warn(f'    - Epoch: {epochi} - Eval Loss: {eval_loss}')

            torch.distributed.barrier()
        else:
            torch.distributed.barrier()

    def preprocess_inputs(self, inputs):
        inputs = {
            key: value.to(self.gpu_id)
            for key, value in inputs.items()
            if key in self._model_input_args
        }
        return inputs

    def hf_save_best_model(self):
        best_model_dir = str(self.training_args.output_dir) + "-best-val"
        self.model.module.save_pretrained(best_model_dir)
        logger.warn(f"    - Best model is saved at {best_model_dir}")
        return

    def hf_save_pretrained_model(self):
        self.model.module.save_pretrained(self.training_args.output_dir)
        return

    @staticmethod
    def get_pt_dataset(dataset, batch_size=16):
        pt_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            sampler=DistributedSampler(dataset)
        )
        return pt_dataloader
