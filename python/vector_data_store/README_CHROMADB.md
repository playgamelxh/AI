## 介绍
Chroma 是一个开源的轻量级向量数据库（Vector Database），专注于为 AI 应用（尤其是大语言模型、检索增强生成 RAG 等场景）提供简单、高效的向量存储与检索能力。它的核心特点是易于部署、API 简洁、开箱即用，非常适合开发者快速集成到各类 AI 应用中。

## 核心特点
1. 轻量级与易用性
无需复杂的配置或依赖，可通过 Python 一行代码启动，支持嵌入式模式（直接在应用进程内运行），适合开发、测试及中小规模生产环境。
2. 完整的向量数据库功能
支持向量存储、相似度检索（欧氏距离、余弦相似度等）、元数据过滤、索引优化等核心能力，满足 RAG 等场景的检索需求。
3. 多语言 API 支持
原生支持 Python SDK，同时提供 HTTP API 供其他语言（如 JavaScript、Java 等）调用，易于跨语言集成。
4. 开源免费
完全开源，基于 MIT 协议，可自由修改和商用，无 licensing 限制。

## 典型应用场景
1. 检索增强生成（RAG）：存储文档的向量嵌入（Embedding），快速检索与用户查询相关的上下文，增强大模型回答的准确性。
2. 语义搜索：基于向量相似度实现跨文本、图像的语义匹配（需先将数据转换为向量）。
3. 推荐系统：存储用户 / 物品的向量特征，通过相似度检索实现个性化推荐。

## 快速上手
1. 安装
```azure
pip install chromadb
# 
```
2. 基础使用示例
```azure
import chromadb
from chromadb.utils import embedding_functions

# 初始化客户端（使用内存模式，数据仅存于内存）
client = chromadb.Client()

# 创建一个集合（Collection，类似数据库中的表）
collection = client.create_collection(name="my_collection")

# 向集合中添加数据（包含文档、ID 和可选的元数据）
collection.add(
    documents=[
        "Chroma 是一个轻量级向量数据库",
        "向量数据库用于存储和检索 AI 模型生成的嵌入向量",
        "RAG 技术通过检索相关文档增强大模型的回答能力"
    ],
    ids=["id1", "id2", "id3"],  # 每个文档的唯一标识
    metadatas=[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]  # 可选元数据
)

# 检索与查询相似的文档（自动使用默认嵌入模型生成查询向量）
results = collection.query(
    query_texts=["什么是向量数据库？"],  # 查询文本
    n_results=2  # 返回最相似的 2 个结果
)

print(results)
```
3. 输出说明
   查询结果包含匹配的文档、ID、相似度分数及元数据，例如：
```
   {
"ids": [["id2", "id1"]],
"distances": [[0.321, 0.567]],  # 距离越小，相似度越高
"documents": [
["向量数据库用于存储和检索 AI 模型生成的嵌入向量", "Chroma 是一个轻量级向量数据库"]
],
"metadatas": [[{"source": "doc2"}, {"source": "doc1"}]]
}
   ```

## 部署方式
1. 嵌入式模式：直接在应用中初始化客户端（chromadb.Client()），数据默认存于内存（可配置本地文件持久化），适合单机场景。
2. 服务端模式：启动独立的 Chroma 服务，通过 HTTP 或 gRPC 远程访问，支持多客户端连接：
```azure
# 启动服务端（默认端口 8000）
chroma run --path ./chroma_data  # 数据持久化到 ./chroma_data 目录

# 客户端连接
client = chromadb.HttpClient(host="localhost", port=8000)
```

## 与其他向量数据库的对比
* 相比 Milvus、Weaviate 等重型向量数据库，Chroma 更轻量，部署和维护成本低，但在大规模数据（亿级以上向量）和高并发场景下性能可能稍弱。
* 相比 Pinecone（云服务），Chroma 开源免费，可私有部署，适合对数据隐私有要求的场景。

## 总结
Chroma 是 AI 开发者的 “入门级向量数据库”，尤其适合快速验证 RAG 等场景的想法，或中小规模应用的向量存储需求。其极简的 API 设计降低了向量数据库的使用门槛，让开发者可以更专注于业务逻辑而非基础设施。
