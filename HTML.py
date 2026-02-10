import pandas as pd
import json
from openai import OpenAI
import os, time
import streamlit as st
from datetime import datetime


# -------------------------- 核心翻译函数（复用你的逻辑） --------------------------
def load_glossary(glossary_content):
    """加载术语表（适配Streamlit文件上传）"""
    try:
        glossary = json.loads(glossary_content)
        if not isinstance(glossary, dict):
            raise ValueError("术语表JSON必须是键值对字典格式（键=日文，值=英文）")
        st.success(f"✅ 成功加载术语表，共{len(glossary)}个翻译规则")
        return glossary
    except json.JSONDecodeError:
        raise ValueError("术语表不是合法的JSON格式，请检查文件内容")
    except Exception as e:
        raise Exception(f"读取术语表失败：{str(e)}")


def translate_column(df, col_name, glossary, target_lang="translated", model="qwen-long"):
    """核心翻译逻辑（保留原有逻辑，增加进度提示）"""
    new_col = f"{col_name}_{target_lang}"
    df[new_col] = ""

    # 构建术语表字符串
    glossary_str = "；".join([f"{k}：{v}" for k, v in glossary.items()])
    sys_prompt = (
        "你是一名香港无印良品的電商翻譯專家，嚴格輸出英文，"
        "禁用簡體字，使用英文标点符号，保留品牌/型號原文。\n"
        "【翻譯強制規則】必須嚴格遵循以下術語表進行翻譯，術語表中的日文原文對應固定英文翻译，全文保持一致：\n"
        f"{glossary_str}\n"
        "若原文未在術語表中，請按香港無印良品電商習慣翻譯，確保語氣正式、符合當地用詞習慣。"
    )

    # 初始化进度条
    progress_bar = st.progress(0)
    total_rows = len(df)

    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    for idx, row in df.iterrows():
        original = row[col_name]

        if pd.isna(original) or str(original).strip() == "":
            df.at[idx, new_col] = ""
            progress_bar.progress((idx + 1) / total_rows)
            continue

        user_prompt = f"日文原文：{original}\n香港英文翻譯："

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                top_p=0.5
            )
            df.at[idx, new_col] = resp.choices[0].message.content.strip()
            time.sleep(0.3)
        except Exception as e:
            error_msg = f"[ERROR: {str(e)[:200]}]"
            df.at[idx, new_col] = error_msg
            st.warning(f"⚠️ 行 {idx} 翻译失败: {str(e)[:100]}")

        # 更新进度条
        progress_bar.progress((idx + 1) / total_rows)

    progress_bar.empty()  # 完成后清空进度条
    return df


# -------------------------- Streamlit前端界面 --------------------------
def main():
    # 页面基础配置
    st.set_page_config(
        page_title="无印良品电商翻译工具",
        page_icon="📝",
        layout="wide"
    )

    # 标题和说明
    st.title("📝 无印良品电商日文→香港英文翻译工具")
    st.markdown("---")
    st.subheader("使用说明")
    st.markdown("""
    1. 确保已配置环境变量 `DASHSCOPE_API_KEY`（通义千问API密钥）
    2. 上传需要翻译的Excel文件（仅处理指定列）
    3. 上传术语表JSON文件（键=日文，值=英文）
    4. 选择待翻译列名，点击开始翻译
    5. 翻译完成后下载结果文件
    """)
    st.markdown("---")

    # 左侧：文件上传和配置
    with st.sidebar:
        st.header("⚙️ 上传与配置")

        # 1. 上传Excel文件
        excel_file = st.file_uploader("📂 上传待翻译的Excel文件", type=["xlsx"])

        # 2. 上传术语表JSON文件
        glossary_file = st.file_uploader("📑 上传术语表JSON文件", type=["json"])

        # 3. 配置项
        if excel_file:
            # 读取Excel并显示可选列名
            df_sample = pd.read_excel(excel_file)
            col_name = st.selectbox("🔤 选择待翻译的列名", df_sample.columns)

        # 4. 模型选择
        model = st.selectbox("🤖 选择翻译模型", ["qwen-long", "qwen-turbo"], index=0)

        # 5. 开始翻译按钮
        translate_btn = st.button("🚀 开始翻译", type="primary", disabled=not (excel_file and glossary_file))

    # 右侧：结果展示和下载
    st.header("📊 翻译结果")
    result_placeholder = st.empty()

    # 核心逻辑：点击翻译按钮后的处理
    if translate_btn:
        try:
            # 1. 读取上传的文件
            with st.spinner("📤 正在读取文件..."):
                df = pd.read_excel(excel_file)
                glossary_content = glossary_file.getvalue().decode("utf-8")
                glossary = load_glossary(glossary_content)

            # 2. 执行翻译
            with st.spinner("🔍 正在翻译中，请稍候..."):
                df_translated = translate_column(
                    df=df,
                    col_name=col_name,
                    glossary=glossary,
                    target_lang="translated",
                    model=model
                )

            # 3. 展示结果
            result_placeholder.dataframe(df_translated, use_container_width=True)

            # 4. 生成下载文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"翻译结果_{timestamp}.xlsx"

            # 将DataFrame转为Excel二进制流
            from io import BytesIO
            output = BytesIO()
            df_translated.to_excel(output, index=False, engine="openpyxl")
            output.seek(0)

            # 下载按钮
            st.download_button(
                label="💾 下载翻译结果",
                data=output,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("🎉 翻译完成！结果已准备好下载")

        except Exception as e:
            st.error(f"❌ 处理失败：{str(e)}")


if __name__ == "__main__":
    # 检查API密钥是否配置
    if not os.getenv("DASHSCOPE_API_KEY"):
        st.warning("⚠️ 未检测到环境变量 DASHSCOPE_API_KEY，请先配置通义千问API密钥！")
    main()