import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.t5_module import T5FineTunerLitModule


def upload_model(model_id, checkpoint):
    finetuner = T5FineTunerLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
    finetuner.model.push_to_hub(model_id)
    finetuner.tokenizer.push_to_hub(model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model to the Hugging Face Hub.")
    parser.add_argument("model_id", type=str, help="The model ID repo to upload.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint.")
    args = parser.parse_args()
    upload_model(args.model_id, args.checkpoint)
