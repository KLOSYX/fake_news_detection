import torch
import torch.nn as nn
from einops import repeat
from med import BertForMaskedLM, BertLMHeadModel
from transformers import (
    BertConfig,
    BertTokenizer,
    CLIPProcessor,
    CLIPVisionModel,
    DataCollatorForLanguageModeling,
    DeiTModel,
    ViTModel,
)


class VisualEncoderLMDecoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        visual_encoder: str = "openai/clip-vit-base-patch32",
        text_decoder: str = "hfl/chinese-roberta-wwm-ext",
    ) -> None:
        super().__init__()
        if "clip" in visual_encoder:
            self.visual_encoder = CLIPVisionModel.from_pretrained(
                visual_encoder, cache_dir="~/.cache"
            )
        elif "deit" in visual_encoder:
            self.visual_encoder = DeiTModel.from_pretrained(visual_encoder, cache_dir="~/.cache")
        else:
            self.visual_encoder = ViTModel.from_pretrained(visual_encoder, cache_dir="~/.cache")
        self.tokenizer = tokenizer

        self.text_decoder_config = BertConfig.from_pretrained(text_decoder, cache_dir="~/.cache")
        self.text_decoder_config.is_decoder = True
        self.text_decoder_config.add_cross_attention = True
        # self.text_decoder_config.vocab_size = len(tokenizer)
        self.text_decoder_config.bos_token_id = (
            tokenizer.enc_token_id
            if hasattr(tokenizer, "enc_token_id")
            else tokenizer.cls_token_id
        )
        self.text_decoder_config.eos_token_id = tokenizer.sep_token_id
        self.text_decoder_config.pad_token_id = tokenizer.pad_token_id
        self.text_decoder_config.encoder_width = self.visual_encoder.config.hidden_size
        self.text_decoder = BertLMHeadModel.from_pretrained(
            pretrained_model_name_or_path=text_decoder, config=self.text_decoder_config
        )

    def forward(self, pixels, text_inputs):
        encoder_outputs = self.visual_encoder(pixel_values=pixels, return_dict=True)
        last_visual_tokens = encoder_outputs.last_hidden_state  # (N, L, dim)
        text_targets = text_inputs.input_ids.masked_fill(
            text_inputs.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_outputs = self.text_decoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            encoder_hidden_states=last_visual_tokens,
            labels=text_targets,
            return_dict=True,
            mode="multimodal",
        )
        loss = decoder_outputs.loss
        return loss

    @torch.no_grad()
    def generate(
        self,
        pixels,
        sample=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        **kwargs
    ):
        bos_token_id = (
            self.tokenizer.cls_token_id
            if not hasattr(self.tokenizer, "enc_token_id")
            else self.tokenizer.enc_token_id
        )
        image_embeds = self.visual_encoder(
            pixel_values=pixels, return_dict=True
        ).last_hidden_state  # [N, L, dim]
        if not sample:
            image_embeds = image_embeds.repeat_interleave(
                num_beams, dim=0
            )  # [num_beams * N, L, dim]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }
        input_ids = torch.tensor([bos_token_id], dtype=torch.long).to(image_embeds.device)
        input_ids = repeat(input_ids, "1 -> N 1", N=pixels.size(0))

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                bos_token_id=bos_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs,
                **kwargs
            )
        else:
            # beam search
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                bos_token_id=bos_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs,
                **kwargs
            )
        return outputs


class VisualEncoderMLMEncoder(nn.Module):
    def __init__(
        self,
        visual_encoder: str = "openai/clip-vit-base-patch32",
        text_decoder: str = "hfl/chinese-roberta-wwm-ext",
        momentum=0.05,
        queue_size=16380,
    ) -> None:
        super().__init__()
        if "clip" in visual_encoder:
            self.visual_encoder = CLIPVisionModel.from_pretrained(
                visual_encoder, cache_dir="~/.cache"
            )
        else:
            self.visual_encoder = ViTModel.from_pretrained(visual_encoder, cache_dir="~/.cache")

        self.text_decoder_config = BertConfig.from_pretrained(text_decoder, cache_dir="~/.cache")
        # add cross attention layers to interact with visual encoder
        self.text_decoder_config.is_decoder = True
        self.text_decoder_config.add_cross_attention = True
        self.text_decoder_config.encoder_width = self.visual_encoder.config.hidden_size
        self.text_decoder = BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=text_decoder, config=self.text_decoder_config
        )
        # self.global_repr_proj = nn.Linear(self.text_decoder.config.hidden_size, 256)

    def forward(self, pixel_values, text_inputs, valid_imgs, is_train=True):
        # loss for mlm
        encoder_outputs = self.visual_encoder(pixel_values=pixel_values, return_dict=True)
        last_visual_tokens = encoder_outputs.last_hidden_state  # (N, L, dim)
        sequence_length = last_visual_tokens.size(1)
        # encoder_attention_mask = torch.tensor(valid_imgs).to(torch.float).to(text_inputs.input_ids.device)
        encoder_attention_mask = repeat(valid_imgs, "N -> N L", L=sequence_length)
        outputs = self.text_decoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            encoder_hidden_states=last_visual_tokens,
            encoder_attention_mask=encoder_attention_mask,
            labels=text_inputs.labels if is_train else None,
            return_dict=True,
            output_hidden_states=True,
            mode="multimodal",
        )

        # global_repr = self.global_repr_proj(outputs.hidden_states[-1][:, 0, :]) # (N, 256)
        # global_repr = F.normalize(global_repr, dim=1)
        last_hidden_state = outputs.hidden_states[-1]  # (N, L, dim)

        return (outputs.loss, last_hidden_state) if is_train else last_hidden_state[:, 0, :]


if __name__ == "__main__":
    # debug
    import requests
    from PIL import Image

    # model = VisualEncoderLMDecoder()
    model = VisualEncoderMLMEncoder()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="~/.cache")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir="~/.cache")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, return_tensors="pt")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    text = [
        "猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉",
        "猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉猫猫在睡觉",
    ]
    image = [Image.open(requests.get(url, stream=True).raw)] * 2
    visual_inputs = processor(images=image, return_tensors="pt")
    text_inputs = tokenizer(text, return_tensors="pt", return_special_tokens_mask=True)
    text_inputs_mlm = data_collator([text_inputs])

    loss = model(visual_inputs, text_inputs_mlm, valid_imgs=[True, False])

    generated_ids = model.generate(visual_inputs)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    ...
