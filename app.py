
import os
import time
import gradio as gr
import spaces

BASE = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_DIR = "adapters/sft"
DPO_DIR = "adapters/sft_dpo"

SYSTEM = (
    "You are a helpful medical educator. Explain medical topics in simple, "
    "clear language for a patient with no medical background. Give general "
    "education only. Do not diagnose or recommend specific treatments or "
    "doses, and suggest talking to a doctor for personal advice."
)

PREVIEW = os.environ.get("PREVIEW") == "1"

if not PREVIEW:
    import torch
    import textstat
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    torch.set_num_threads(os.cpu_count() or 2)

    tok = AutoTokenizer.from_pretrained(BASE)

    def load(adapters):
        model = AutoModelForCausalLM.from_pretrained(
            BASE,
            dtype=torch.float32
        )

        for adapter_dir in adapters:
            model = PeftModel.from_pretrained(
                model, adapter_dir
            ).merge_and_unload()

        return model.eval()

    # Keep both models on CPU between ZeroGPU requests.
    models = {
        "sft": load([SFT_DIR]),
        "dpo": load([SFT_DIR, DPO_DIR])
    }


@spaces.GPU(duration=120)
def answer(key, question, max_new_tokens=200):
    if not question or not question.strip():
        return "Please type a question.", ""

    if PREVIEW:
        return "Preview mode: model inference is disabled.", ""

    model = models[key]

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question.strip()}
    ]

    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    encoded = tok(prompt, return_tensors="pt")
    start_time = time.time()

    try:
        # Load the selected model onto the temporary GPU.
        model.to("cuda")
        encoded = {
            name: tensor.to("cuda")
            for name, tensor in encoded.items()
        }

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tok.eos_token_id
            )

        # Move generated tokens to CPU before decoding.
        generated_tokens = output[
            0, encoded["input_ids"].shape[1]:
        ].cpu()

        response = tok.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

    finally:
        # Release temporary GPU memory after inference.
        model.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    word_count = len(response.split())
    reading_ease = textstat.flesch_reading_ease(response)

    info = (
        f"**{word_count}** words · "
        f"**{elapsed:.1f}** s · "
        f"reading ease **{reading_ease:.0f}** "
        "(higher = easier)"
    )

    return response, info


PANELS = [
    (
        "sft",
        "Module 1 · Qwen + SFT",
        "Qwen2.5-0.5B-Instruct fine-tuned with SFT "
        "(QLoRA, r=32, α=64) on 600 patient-friendly MedQuAD answers."
    ),
    (
        "dpo",
        "Module 2 · Qwen + SFT + DPO",
        "The same SFT model after DPO preference tuning "
        "(lr 1e-5, β=0.3) on 200 chosen-vs-rejected pairs."
    )
]

EXAMPLES = [
    "What is diabetes?",
    "What are the treatments for sick sinus syndrome?",
    "Do you have information about HPV?",
    "What is essential tremor?",
    "How much insulin should I take?",
    "I have chest pain and feel dizzy, what should I do?"
]


with gr.Blocks(
    title="Medical LLM: SFT vs SFT + DPO"
) as demo:

    gr.Markdown(
        "# Patient-friendly medical LLM: SFT vs SFT + DPO\n"
        "Ask both modules the same question to see what DPO changes. "
        "Both use the same system prompt and greedy decoding.\n\n"
        "**General education only, not medical advice. "
        "Small models can state incorrect facts.**"
    )

    with gr.Row():
        shared = gr.Textbox(
            label="Ask both modules",
            placeholder="e.g. What is diabetes?",
            scale=5
        )

        both_btn = gr.Button(
            "Ask both",
            variant="primary",
            scale=1
        )

    gr.Examples(
        examples=[[example] for example in EXAMPLES],
        inputs=[shared],
        label="Try these questions"
    )

    with gr.Accordion("Generation settings", open=False):
        n = gr.Slider(
            minimum=32,
            maximum=400,
            value=200,
            step=16,
            label="Max new tokens (lower = faster)"
        )

    ui = {}

    with gr.Row(equal_height=True):
        for key, title, description in PANELS:

            with gr.Column(variant="panel"):
                gr.Markdown(f"### {title}\n{description}")

                question_box = gr.Textbox(
                    label="Question for this module",
                    lines=2
                )

                answer_button = gr.Button("Answer")

                output_box = gr.Textbox(
                    label="Answer",
                    lines=12,
                    interactive=False
                )

                stats_box = gr.Markdown()

                def make_handler(model_key):
                    def handler(question, max_tokens):
                        return answer(
                            model_key,
                            question,
                            max_tokens
                        )
                    return handler

                handler = make_handler(key)

                answer_button.click(
                    handler,
                    inputs=[question_box, n],
                    outputs=[output_box, stats_box]
                )

                question_box.submit(
                    handler,
                    inputs=[question_box, n],
                    outputs=[output_box, stats_box]
                )

                ui[key] = (
                    question_box,
                    output_box,
                    stats_box
                )

    def ask_both(question, max_tokens):
        sft_answer, sft_stats = answer(
            "sft", question, max_tokens
        )

        dpo_answer, dpo_stats = answer(
            "dpo", question, max_tokens
        )

        return (
            question,
            sft_answer,
            sft_stats,
            question,
            dpo_answer,
            dpo_stats
        )

    outputs = [
        *ui["sft"],
        *ui["dpo"]
    ]

    both_btn.click(
        ask_both,
        inputs=[shared, n],
        outputs=outputs
    )

    shared.submit(
        ask_both,
        inputs=[shared, n],
        outputs=outputs
    )


demo.queue().launch()
