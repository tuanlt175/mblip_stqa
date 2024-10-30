"""Module đánh giá model sửa lỗi chính tả."""

import argparse
import tqdm
from icvrc import VRC
from icvrc.scoring.evaluation import compute_scores
from icvrc.utils import normalize
import json
import os


def main():
    """Hàm chính để chạy đánh giá model."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model",
        type=str,
        help="Đường dẫn đến thư mục chứa model",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="Đường dẫn đến file json chứa dữ liệu test.",
    )
    parser.add_argument(
        "--images_folder",
        type=str,
        help="Đường dẫn đến thư mục chứa ảnh.",
    )
    parser.add_argument(
        "--submission_file",
        default=None,
        type=str,
        help="Đường dẫn đến file json chứa câu trả lời của model.",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Đường dẫn đến thư mục chứa kết quả đánh giá.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--question_max_len",
        type=int,
        default=48,
        help="Chiều dài tối đa của câu hỏi.",
    )
    parser.add_argument(
        "--answer_max_len",
        type=int,
        default=64,
        help="Chiều dài tối đa của câu trả lời.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam size.",
    )

    args = parser.parse_args()

    with open(args.data_file, "r") as file:
        data = json.load(file)

    data = [
        {
            "id": id,
            "image_file": os.path.join(args.images_folder, data["images"][str(item["image_id"])]),
            "question": item["question"],
            "answer": item["answer"],
            "image_text": item.get("image_text", ""),
        }
        for id, item in data["annotations"].items()
    ]

    model = VRC(
        args.model,
        question_max_len=args.question_max_len,
        answer_max_len=args.answer_max_len,
    )

    for start in tqdm.tqdm(range(0, len(data), args.batch_size)):
        end = min(len(data), start + args.batch_size)
        pred_answers = model.batch_infer(data[start:end], num_beams=args.num_beams)
        for item, answer in zip(data[start:end], pred_answers):
            item["pred_answer"] = answer

    gts = {x["id"]: [normalize(x["answer"])] for x in data}
    gens = {x["id"]: [x["pred_answer"]] for x in data}

    results = compute_scores(gts, gens)
    output_log = f"## Model: {args.model} ##"
    output_log += "\n    - Số dữ liệu: " + str(len(data))
    output_log += "\n    - BLEU: " + str(round(results["BLEU"], 4)).replace(".", ",")
    output_log += "\n    - CIDEr: " + str(round(results["CIDEr"], 4)).replace(".", ",")
    print(output_log)

    if isinstance(args.output_folder, str):
        output_file = "-".join(args.model.strip("/").split("/")[-2:]) + ".txt"
        output_file = os.path.join(args.output_folder, output_file)
        with open(output_file, "w") as file:
            file.write(output_log)

    if isinstance(args.submission_file, str):
        with open(args.submission_file, "w") as file:
            json.dump(
                {x["id"]: x["pred_answer"] for x in data},
                file, ensure_ascii=False, indent=4
            )


if __name__ == "__main__":
    main()
