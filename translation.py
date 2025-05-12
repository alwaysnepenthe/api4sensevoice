from transformers import pipeline as hf_pipeline

lang_mapping = {
    "bg": "bul_Cyrl",  # 保加利亚语
    "hr": "hrv_Latn",  # 克罗地亚语
    "cs": "ces_Latn",  # 捷克语
    "da": "dan_Latn",  # 丹麦语
    "nl": "nld_Latn",  # 荷兰语
    "en": "eng_Latn",  # 英语
    "et": "est_Latn",  # 爱沙尼亚语
    "fi": "fin_Latn",  # 芬兰语
    "fr": "fra_Latn",  # 法语
    "de": "deu_Latn",  # 德语
    "el": "ell_Grek",  # 希腊语
    "hu": "hun_Latn",  # 匈牙利语
    "ga": "gle_Latn",  # 爱尔兰语
    "it": "ita_Latn",  # 意大利语
    "lv": "lvs_Latn",  # 拉脱维亚语
    "lt": "lit_Latn",  # 立陶宛语
    "mt": "mlt_Latn",  # 马耳他语
    "pl": "pol_Latn",  # 波兰语
    "pt": "por_Latn",  # 葡萄牙语
    "ro": "ron_Latn",  # 罗马尼亚语
    "sk": "slk_Latn",  # 斯洛伐克语
    "sl": "slv_Latn",  # 斯洛文尼亚语
    "es": "spa_Latn",  # 西班牙语
    "sv": "wes_Latn",  # 瑞典语
    "ru": "rus_Cyrl",  # 俄语
    "tr": "tur_Latn",  # 土耳其语
    "eu": "eus_Latn",  # 巴斯克语
    "ca": "cat_Latn",  # 加泰罗尼亚语
    "sq": "sqi_Latn",  # 阿尔巴尼亚语
    "se": "smj_Latn",  # 北萨米语
    "uk": "ukr_Cyrl",  # 乌克兰语
    "no": "nob_Latn",  # 挪威语
    "ar": "arb_Arab",  # 阿拉伯语
    "zh": "zho_Hans",  # 中文（简体）
    "he": "heb_Hebr"   # 希伯来语
}

#完成语言类别识别
model_langreco = hf_pipeline(
    "text-classification",
    model = "ERCDiDip/langdetect"
)

##完成翻译功能
model_trans = hf_pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
)

#完成语言类别识别
def lang_reco(text):
    try:
        res = model_langreco(text)
        src_lang = lang_mapping.get(res[0]['label'],"eng_Latn")
        print(f"text: {text} ; src_lang: {src_lang}");
        return src_lang
    except Exception as e:
        print(f"Error in lang_reco function: {e}")
        return "auto"

#完成翻译
def translate_text(text,src_lang,tgt_lang):
    try:
        print("test translate_text")
        res= model_trans(text,src_lang = src_lang,tgt_lang = tgt_lang);
        return res[0]['translation_text']
    except Exception as e:
        print(f"Error in translate_text function: {e}")
        return text
    
def main():
    text = "Bonjour"
    src_lang = lang_reco(text)
    tgt_lang = "zho_Hans"
    translated_text = translate_text(text, src_lang, tgt_lang)
    print(f"Translated text: {translated_text}")

if __name__ == "__main__":
    main()
