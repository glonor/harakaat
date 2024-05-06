import gradio as gr
from transformers import pipeline

examples = ["أنا أذهب كل سبت إلى المدرسة لأدرس اللغة العربية مع أصدقائي", "شكرا على المساعدة", "عفوا، لا أستطيع"]


generator = pipeline("text2text-generation", model="glonor/byt5-arabic-diacritization")


def diacratize_text(text):
    return generator(text, max_length=512)[0]["generated_text"]


interface = gr.Interface(
    fn=diacratize_text,
    inputs=gr.Textbox(lines=3, placeholder="Enter Arabic Text Here...", label="Input text", text_align="right"),
    outputs=gr.Textbox(label="Diacratized Text", text_align="right", show_copy_button=True),
    title="Arabic Text Diacritization",
    allow_flagging="never",
    examples=examples,
    cache_examples=True,
)
interface.launch()
