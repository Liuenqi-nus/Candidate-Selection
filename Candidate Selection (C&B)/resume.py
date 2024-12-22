import os
import glob
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


########################################
# 1. 读取简历文本
########################################

def load_resumes_from_folder(folder_path: str):
    """
    读取指定文件夹下所有 .txt 文件，返回文本列表和对应的文件名列表
    :param folder_path: 简历所在的文件夹路径
    :return:
        texts: list[str], 每个元素是一份简历的文本内容
        filenames: list[str], 对应的简历文件名（如 'resume_01.txt')
    """
    resume_files = glob.glob(os.path.join(folder_path, "*.txt"))
    texts = []
    filenames = []

    for f in resume_files:
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
            texts.append(text)
            filenames.append(os.path.basename(f))
    return texts, filenames


########################################
# 2. 解析结构化信息 (Mock 示例)
########################################

def parse_structured_info_from_text(resume_text):
    """
    一个非常简化的示例函数，用于从简历文本中提取一些关键字段。
    实际项目中通常要使用专业的简历解析工具/算法。

    返回一个结构化信息字典，如:
    {
      "highest_education": "Master",  # or "Bachelor", "PhD", ...
      "years_of_experience": 5,
      "certifications": ["SHRM-SCP", "CCWP"],
      ...
    }
    """
    structured_data = {
        "highest_education": None,
        "years_of_experience": 0,
        "certifications": []
    }

    # 示例1：最高学历匹配 (简单用正则搜几个关键词)
    # 注意，这只是演示，真实场景要更可靠的正则/解析逻辑
    education_pattern = re.compile(r"(phd|doctor|master|bachelor)", re.IGNORECASE)
    edu_match = education_pattern.search(resume_text)
    if edu_match:
        # 将捕获到的学历单词统一成固定格式
        word = edu_match.group(1).lower()
        if word in ["phd", "doctor"]:
            structured_data["highest_education"] = "PhD"
        elif word == "master":
            structured_data["highest_education"] = "Master"
        elif word == "bachelor":
            structured_data["highest_education"] = "Bachelor"
    else:
        structured_data["highest_education"] = "Unknown"

    # 示例2：工作年限 (非常粗糙的正则 + 提取)
    # 例如 “5 years experience”、“3+ years”、“2 yrs”等
    exp_pattern = re.compile(r"(\d+)\s*(\+?)(?:\s*years|\s*yrs)", re.IGNORECASE)
    exp_match = exp_pattern.search(resume_text)
    if exp_match:
        years_str = exp_match.group(1)  # 抓到数字
        plus_sign = exp_match.group(2)  # 如果有 '+'
        years = int(years_str)
        if plus_sign == "+":
            years += 1  # 简单处理下带+的情况
        structured_data["years_of_experience"] = years

    # 示例3：证书 (简单示例：SHRM|CCWP|PHR等)
    cert_pattern = re.compile(r"(SHRM-SCP|SHRM-CP|CCWP|PHR|SPHR)", re.IGNORECASE)
    certs_found = cert_pattern.findall(resume_text)
    # 去重 & 格式化
    certs_cleaned = list(set(c.strip().upper() for c in certs_found))
    structured_data["certifications"] = certs_cleaned

    return structured_data


########################################
# 3. 基于规则的打分 (Rule-based Scoring)
########################################

def get_rule_based_score(struct_info):
    """
    根据结构化信息进行简单的加权或加分扣分。
    这里仅做演示示例，你可以自行调整具体规则和权重。
    """
    score = 0.0

    # 学历打分示例
    edu = struct_info.get("highest_education", "Unknown")
    if edu == "PhD":
        score += 10
    elif edu == "Master":
        score += 6
    elif edu == "Bachelor":
        score += 3

    # 工作年限打分示例
    years = struct_info.get("years_of_experience", 0)
    # 假设想要至少 3 年经验：每超出 1 年 +1 分，低于 3 年就扣分
    if years >= 3:
        score += (years - 3)  # 例如 5年→ +2分
    else:
        score -= (3 - years)  # 1年→ -2分

    # 证书加分示例
    certs = struct_info.get("certifications", [])
    for c in certs:
        if c in ["SHRM-SCP", "SPHR"]:
            score += 5
        elif c in ["CCWP", "PHR"]:
            score += 3

    return score


########################################
# 4. 基于 TF-IDF 的文本打分 (ML / Semantic Scoring)
########################################

def rank_by_tfidf_score(resume_texts, query):
    """
    传入所有简历文本和查询，返回每份简历与查询的相似度分数列表。
    （与后面的规则分数是相独立的）
    """

    # tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(resume_texts)
    query_vec = tfidf_vectorizer.transform([query])

    dot_products = tfidf_matrix.dot(query_vec.T).toarray().ravel()
    query_norm = np.linalg.norm(query_vec.toarray())
    resume_norms = np.linalg.norm(tfidf_matrix.toarray(), axis=1)
    eps = 1e-9
    cos_sim = dot_products / (resume_norms * query_norm + eps)

    return cos_sim  # 返回一个与 resume_texts 等长的向量，每个元素是对应简历的 TF-IDF 相似度分值


########################################
# 5. 主函数：融合规则分数 + 文本分数
########################################

def hybrid_score_ranking(resume_texts, resume_names, query, alpha=0.5):
    """
    根据 "规则分数" 和 "文本相似度分数" 进行混合打分。
    :param resume_texts: 所有简历的非结构化文本
    :param resume_names: 对应的简历文件名
    :param query: 查询/岗位需求描述
    :param alpha: 融合系数，决定规则分数与文本分数的权重
                  total_score = alpha * rule_score + (1 - alpha) * text_score
    :return: 按照混合分数排序的结果
    """
    # Step A: 基于 TF-IDF 的文本相似度分数
    text_scores = rank_by_tfidf_score(resume_texts, query)

    # Step B: 基于规则的结构化打分
    rule_scores = []
    for text in resume_texts:
        struct_info = parse_structured_info_from_text(text)
        rule_score = get_rule_based_score(struct_info)
        rule_scores.append(rule_score)

    # 为了让二者在同一量级，可以做简单的归一化处理
    # 先转为 numpy array
    text_scores = np.array(text_scores)
    rule_scores = np.array(rule_scores)

    # 归一化到 [0,1]，也可以用别的方式
    # 如果存在负值，需要先平移或选用别的归一化策略
    def min_max_scale(arr):
        min_val, max_val = arr.min(), arr.max()
        if abs(max_val - min_val) < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    text_scores_norm = min_max_scale(text_scores)
    rule_scores_norm = min_max_scale(rule_scores)

    # Step C: 融合分数
    # total_score = alpha * rule_score_norm + (1 - alpha) * text_score_norm
    total_scores = alpha * rule_scores_norm + (1 - alpha) * text_scores_norm

    # Step D: 按照 total_score 降序排序
    ranked_indices = np.argsort(-total_scores)

    # Step E: 组织输出
    results = []
    for idx in ranked_indices:
        results.append({
            "resume_name": resume_names[idx],
            "rule_score_raw": float(rule_scores[idx]),
            "text_score_raw": float(text_scores[idx]),
            "rule_score_norm": float(rule_scores_norm[idx]),
            "text_score_norm": float(text_scores_norm[idx]),
            "total_score": float(total_scores[idx])
        })
    return results


########################################
# 6. 测试入口
########################################

if __name__ == "__main__":
    # 1. 读取本地简历
    resume_folder = "data"
    all_texts, all_names = load_resumes_from_folder(resume_folder)

    # 2. 设定查询/职位描述
    user_query = "candidates applying for senior compensation and benefits role in the Finance industry"

    # 3. 进行混合打分排名
    #   alpha 决定规则分数在总分中占比，可根据实际情况调整
    ranking_results = hybrid_score_ranking(all_texts, all_names, user_query, alpha=0.6)

    # 4. 查看排名结果（此处打印前 10）
    print("=== Hybrid Ranking Results (Top 10) ===")
    for i, item in enumerate(ranking_results[:10]):
        print(f"Rank {i + 1}: {item['resume_name']} | "
              f"RuleRaw={item['rule_score_raw']:.2f}, "
              f"TextRaw={item['text_score_raw']:.2f}, "
              f"RuleNorm={item['rule_score_norm']:.2f}, "
              f"TextNorm={item['text_score_norm']:.2f}, "
              f"Total={item['total_score']:.2f}")
