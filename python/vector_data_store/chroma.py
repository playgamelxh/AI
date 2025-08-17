import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# 初始化客户端（使用内存模式，数据仅存于内存）
client = chromadb.Client(
    Settings(
        persist_directory="./chroma_data",  # 数据存储的本地目录路径
        is_persistent=True  # 启用持久化（默认 True，可省略）
    )
)

# # 创建一个集合（Collection，类似数据库中的表）
# collection = client.create_collection(name="my_collection")

# 加载已存在的集合
collection = client.get_collection(name="my_collection")

#
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

collection.add(
documents=[
"Chroma 是一个轻量级向量数据库",
"向量数据库用于存储和检索 AI 模型生成的嵌入向量",
"RAG 技术通过检索相关文档增强大模型的回答能力"
],
ids=["id4", "id5", "id6"],  # 每个文档的唯一标识
metadatas=[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]  # 可选元数据
)

# # 检索与查询相似的文档（自动使用默认嵌入模型生成查询向量）
# results = collection.query(
# query_texts=["什么是向量数据库？"],  # 查询文本
# n_results=2  # 返回最相似的 2 个结果
# )
#
# print(results)

# 加载已存在的集合
results = collection.query(query_texts=["什么是向量数据库？"], n_results=2)
print(results)

# 本地路径
# /Users/bytedance/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz
