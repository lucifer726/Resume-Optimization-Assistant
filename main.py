import json

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import openai
from langchain_community.document_loaders.pdf import PyPDFLoader

openai.api_key = 'sk-xxxxx'

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 选择一个支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def analyze_resume_with_gpt(resume_text, job_description):
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user",
             "content": f"你是一个专业的面试官，现在有一份候选人的简历: {resume_text}, 还有一个岗位描述: {job_description}。" +
                        "请你根据这8个维度进行评价，并且返回一个json，json的key为我指定的8个维度，value是每个维度的0-10的打分，下面是返回结果的样例：" +
                        "{\"技能匹配度\": score, \"经验相关性\": score, \"教育背景\": score, \"成就与贡献\": score, \"文化适应性\": score, \"职业目标\": score, \"关键词匹配\": score, \"格式和专业性\": score}"}
        ]
    )
    # 提取模型返回的内容
    model_response_content = response.choices[0].message.content

    # 将提取的内容解析为JSON对象
    scores_json = json.loads(model_response_content)

    # 定义评分字典
    scores = {
        "技能匹配度": 0,
        "经验相关性": 0,
        "教育背景": 0,
        "成就与贡献": 0,
        "文化适应性": 0,
        "职业目标": 0,
        "关键词匹配": 0,
        "格式和专业性": 0,
    }

    # 更新scores字典中的评分
    for key in scores.keys():
        if key in scores_json:
            scores[key] = scores_json[key]

    # 返回最终评分结果
    return scores

def optimize_resume_with_gpt(resume_text, job_description):
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user",
             "content": f"你是一个专业的面试官，现在有一份候选人的简历: {resume_text}, 还有一个岗位描述: {job_description}。" +
                        "请针对候选人的简历，从以下几个方面给出优化建议，让他的简历和该岗位更匹配。技能匹配度、经验相关性、教育背景、成就与贡献、文化适应性、职业目标、关键词匹配、格式和专业性。" +
                        "需要给出修改前后修改后的对比，优化建议数量不限，用如下json返回：{\"优化建议1\": 优化结果,\"优化建议2\": 优化结果,\"优化建议3\": 优化结果}"
             }
        ]
    )
    # 提取模型返回的内容
    response_content = response.choices[0].message.content
    response_json = json.loads(response_content)
    print(response_json)

    # 返回最终优化结果
    return response_json

def mock_interview_with_gpt(resume_text, job_description):
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user",
             "content": f"你是一个专业的面试官，现在有一份候选人的简历: {resume_text}, 还有一个岗位描述: {job_description}。" +
                        "请针对候选人的简历，请给出10道面试题，需要结合他的工作经历以及岗位描述，考察他是否对该岗位的技术栈熟练掌握，是否能够胜任该岗位" +
                        "你的问题用如下json返回：{\"问题1\": 内容,\"问题2\": 内容,\"问题3\": 内容, }"
             }
        ]
    )
    # 提取模型返回的内容
    response_content = response.choices[0].message.content
    response_json = json.loads(response_content)
    print(response_json)

    # 返回最终优化结果
    return response_json

def create_radar_chart(scores):
    # 生成能力图的函数，现在添加了分数标注
    labels = np.array(list(scores.keys()))
    stats = np.array(list(scores.values()))

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='blue', alpha=0.25)
    ax.plot(angles, stats, color='blue', linewidth=2)  # Draw the outline of our data.
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # 修改标签以包含分数
    label_with_scores = [f'{label} {score}分' for label, score in zip(labels, stats)]
    ax.set_xticklabels(label_with_scores)

    return fig


def generate_match_chart(pdf_file, job_description):
    loader = PyPDFLoader(pdf_file.name)
    result = loader.load()
    page = result[0]
    resume_text = page.page_content

    scores = analyze_resume_with_gpt(resume_text, job_description)
    fig = create_radar_chart(scores)
    plt.close(fig)  # 关闭图表，防止其显示在notebook中

    # 将图表保存为图片并返回
    radar_chart_path = "radar_chart.png"
    fig.savefig(radar_chart_path)

    return radar_chart_path

def generate_resume_suggestions(pdf_file, job_description):
    loader = PyPDFLoader(pdf_file.name)
    result = loader.load()
    page = result[0]
    resume_text = page.page_content

    suggestion = optimize_resume_with_gpt(resume_text, job_description)

    return suggestion

def generate_mock_interview(pdf_file, job_description):
    loader = PyPDFLoader(pdf_file.name)
    result = loader.load()
    page = result[0]
    resume_text = page.page_content

    interview = mock_interview_with_gpt(resume_text, job_description)

    return interview

def app():
    with gr.Blocks() as demo:
        with gr.Row():
            file_input = gr.File(label="上传PDF简历")
            job_desc_input = gr.Textbox(label="输入目标岗位描述")

        with gr.Row():
            generate_chart_button = gr.Button("生成简历匹配度能力图")
            generate_suggestions_button = gr.Button("生成简历优化建议")
            generate_interview_button = gr.Button("生成模拟面试题")

        with gr.Row():
            output_image = gr.Image(label="简历匹配度能力图")
            output_suggestions_json = gr.Json(label="简历优化建议")
            output_interview_json = gr.Json(label="模拟面试题")

        generate_chart_button.click(
            fn=generate_match_chart,
            inputs=[file_input, job_desc_input],
            outputs=output_image
        )

        generate_suggestions_button.click(
            fn=generate_resume_suggestions,
            inputs=[file_input, job_desc_input],
            outputs=output_suggestions_json
        )

        generate_interview_button.click(
            fn=generate_mock_interview,
            inputs=[file_input, job_desc_input],
            outputs=output_interview_json
        )

    return demo


# 运行Gradio界面
app().launch()