import string

import torch
import streamlit as st
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 模拟早期 RBMT：小型预定义英汉词典（键统一为小写，便于匹配）
RBMT_EN_ZH: dict[str, str] = {
    "i": "我",
    "you": "你",
    "he": "他",
    "she": "她",
    "we": "我们",
    "they": "他们",
    "it": "它",
    "this": "这",
    "that": "那",
    "the": "这",
    "a": "一个",
    "an": "一个",
    "is": "是",
    "are": "是",
    "am": "是",
    "was": "是",
    "were": "是",
    "not": "不",
    "and": "和",
    "or": "或",
    "but": "但是",
    "if": "如果",
    "then": "那么",
    "today": "今天",
    "tomorrow": "明天",
    "yesterday": "昨天",
    "good": "好",
    "bad": "坏",
    "big": "大",
    "small": "小",
    "book": "书",
    "water": "水",
    "sky": "天空",
    "sun": "太阳",
    "moon": "月亮",
    "day": "天",
    "night": "夜",
    "morning": "早晨",
    "time": "时间",
    "year": "年",
    "go": "去",
    "come": "来",
    "see": "看见",
    "make": "做",
    "know": "知道",
    "think": "想",
    "want": "要",
    "like": "喜欢",
    "can": "能",
    "will": "将",
    "rain": "下雨",
    "rains": "下雨",
    "raining": "下雨",
    "cat": "猫",
    "cats": "猫",
    "dog": "狗",
    "dogs": "狗",
    "house": "房子",
    "school": "学校",
    "love": "爱",
    "world": "世界",
    "english": "英语",
    "chinese": "中文",
    "machine": "机器",
    "translation": "翻译",
}


def rule_based_literal_translate(sentence: str, en_zh: dict[str, str] | None = None) -> str:
    """
    模拟基于规则的机器翻译：按空格分词，查词典逐词替换为中文；
    查不到的词保留原文（含大小写与标点附着形式）。
    """
    table = en_zh if en_zh is not None else RBMT_EN_ZH
    sentence = (sentence or "").strip()
    if not sentence:
        return ""

    def lookup_token(raw: str) -> str:
        stripped = raw.strip(string.punctuation)
        if not stripped:
            return raw
        key = stripped.lower()
        if key in table:
            zh = table[key]
            # 若原词带首尾标点，把标点贴回译文两侧（简单处理）
            lead = raw[: raw.index(stripped)] if stripped in raw else ""
            trail = raw[raw.index(stripped) + len(stripped) :] if stripped in raw else ""
            return f"{lead}{zh}{trail}"
        return raw

    return " ".join(lookup_token(tok) for tok in sentence.split())


@st.cache_resource
def load_en_zh_translator():
    """不依赖已移除的 pipeline('translation')，直接用 Marian 编解码。"""
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def translate_en_zh(text: str, tokenizer, model, device: torch.device) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_length=512)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def zh_tokens_for_bleu(text: str) -> list[str]:
    """中文无空格时用语字作为 n-gram 单元，便于 NLTK sentence_bleu。"""
    s = (text or "").strip().replace(" ", "").replace("\n", "").replace("\t", "")
    return list(s) if s else []


def compute_sentence_bleu(reference_zh: str, candidate_zh: str) -> float:
    ref_t = zh_tokens_for_bleu(reference_zh)
    cand_t = zh_tokens_for_bleu(candidate_zh)
    if not ref_t or not cand_t:
        return 0.0
    smooth = SmoothingFunction().method1
    return float(sentence_bleu([ref_t], cand_t, smoothing_function=smooth))


def main():
    st.set_page_config(page_title="多模式英中翻译与评测", layout="centered")

    st.title("多模式英中翻译与评测")
    st.caption("神经机器翻译 · 规则词典直译对比 · BLEU 自动评测")
    st.divider()

    tab_nmt, tab_compare, tab_bleu = st.tabs(
        ["神经机器翻译（英译中）", "对比：NMT 与词典直译", "BLEU 自动评测"]
    )

    with tab_nmt:
        st.header("神经机器翻译（英译中）")
        st.caption("基于 Helsinki-NLP/opus-mt-en-zh（Marian），经 AutoModel 推理")

        user_text = st.text_area("请输入英文句子", height=160, key="nmt_en_input")

        if st.button("确认输入", key="nmt_confirm"):
            text = (user_text or "").strip()
            if not text:
                st.warning("请先输入英文内容。")
            else:
                with st.spinner("正在加载模型并翻译，请稍候…"):
                    tokenizer, model, device = load_en_zh_translator()
                    zh = translate_en_zh(text, tokenizer, model, device)
                st.subheader("中文译文")
                st.write(zh)

    with tab_compare:
        st.header("对比模块：神经翻译 vs 词典逐词直译")
        st.caption(
            "右侧为模拟早期 RBMT：空格分词 + 预定义词典；未命中词条则保留英文。"
        )

        cmp_text = st.text_area("请输入英文句子", height=160, key="cmp_en_input")

        if st.button("生成对比", key="cmp_run"):
            text = (cmp_text or "").strip()
            if not text:
                st.warning("请先输入英文内容。")
            else:
                literal = rule_based_literal_translate(text, RBMT_EN_ZH)
                with st.spinner("正在加载模型并生成神经机器翻译，请稍候…"):
                    tokenizer, model, device = load_en_zh_translator()
                    nmt_zh = translate_en_zh(text, tokenizer, model, device)

                left, right = st.columns(2)
                with left:
                    st.markdown("##### 神经机器翻译（opus-mt）")
                    st.write(nmt_zh)
                with right:
                    st.markdown("##### 基于词典的逐词直译（模拟 RBMT）")
                    st.write(literal)

    with tab_bleu:
        st.header("机器翻译自动评测（BLEU）")
        st.caption(
            "参考译文与候选译文均为中文；BLEU 基于字级 n-gram 重合度（与论文中「分词后 BLEU」略有不同）。"
        )

        st.text_area(
            "1. 待翻译英文原文",
            height=100,
            key="eval_src_en",
            placeholder="用于一键调用模块 1 生成候选译文；仅计算 BLEU 时可留空。",
        )
        st.text_area(
            "2. 标准中文参考译文（Reference）",
            height=120,
            key="eval_ref_zh",
            placeholder="人工标注或可信译文的「标准答案」。",
        )

        gen_c = st.button("使用模块 1（opus-mt）生成候选译文", key="eval_gen_cand")
        if gen_c:
            en = (st.session_state.get("eval_src_en") or "").strip()
            if not en:
                st.warning("请先在「待翻译英文原文」中填写英文，再生成候选译文。")
            else:
                with st.spinner("正在加载模型并生成候选译文…"):
                    tok, mdl, dev = load_en_zh_translator()
                    st.session_state.eval_cand_zh = translate_en_zh(en, tok, mdl, dev)

        st.text_area(
            "3. 机器生成的候选译文（Candidate）",
            height=120,
            key="eval_cand_zh",
            placeholder="可手填，也可点击上方按钮根据英文原文自动生成。",
        )

        if st.button("计算 BLEU", key="eval_bleu_btn"):
            ref = (st.session_state.get("eval_ref_zh") or "").strip()
            cand = (st.session_state.get("eval_cand_zh") or "").strip()
            if not ref or not cand:
                st.warning("请填写参考译文与候选译文后再计算。")
            else:
                score = compute_sentence_bleu(ref, cand)
                st.metric(
                    label="BLEU 得分（NLTK sentence_bleu，字级，平滑 method1）",
                    value=f"{score:.4f}",
                    help="取值约 0～1，越大表示与参考译文的 n-gram 重合度越高。",
                )
                st.caption(f"约合百分制展示：{score * 100:.2f} / 100（非官方换算，仅便于直观理解）")
                st.markdown(
                    """
**这个分数代表什么？**

- **BLEU** 衡量的是：候选译文与参考译文在 **1～4-gram** 上的 **精确匹配比例**（并带 **简短惩罚**），本质是 **字符串层面的重叠度**，**不是**语义等价或「译得对不对」的人工判断。
- **越高**：通常表示候选与参考在 **用词与语序** 上越接近；适合 **同一参考** 下比较多个系统或多次调参的 **相对优劣**。
- **局限**：同义改写、词序合理但与参考字面不同，分数可能偏低；极短句上分数波动大。本页对中文采用 **字级** 切分，与「先分词再算 BLEU」的论文设置可能不同，**适合课堂演示**，跨论文对比数值时需谨慎。
                    """
                )


if __name__ == "__main__":
    main()
