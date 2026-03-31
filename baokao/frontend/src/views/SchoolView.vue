<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const API_BASE = 'http://localhost:8080/api'
const schools = ref([])
const searchCity = ref('')
const searchCategory = ref('')
const dialogVisible = ref(false)
const currentSchool = ref(null)

const fetchSchools = async () => {
  try {
    const res = await axios.get(`${API_BASE}/schools`, {
      params: { city: searchCity.value, category: searchCategory.value }
    })
    schools.value = res.data.data
  } catch (error) {
    ElMessage.error('获取学校列表失败')
  }
}

const handleDetail = async (row) => {
  try {
    const res = await axios.get(`${API_BASE}/schools/${row.id}`)
    currentSchool.value = res.data.data
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error('获取学校详情失败')
  }
}

onMounted(() => {
  fetchSchools()
})

const handleSearch = () => {
  fetchSchools()
}
</script>

<template>
  <div class="school-container">
    <div class="header-action">
      <el-input v-model="searchCity" placeholder="按城市搜索" style="width: 200px; margin-right: 10px;" />
      <el-select v-model="searchCategory" placeholder="按类别搜索" style="width: 200px; margin-right: 10px;" clearable>
        <el-option label="综合类" value="综合类" />
        <el-option label="理工类" value="理工类" />
        <el-option label="艺术类" value="艺术类" />
      </el-select>
      <el-button type="primary" @click="handleSearch">搜索</el-button>
    </div>
    
    <el-table :data="schools" style="width: 100%" border>
      <el-table-column prop="id" label="ID" width="80" />
      <el-table-column prop="name" label="学校名称" width="200" />
      <el-table-column prop="city" label="城市" width="100" />
      <el-table-column prop="level" label="办学层次" width="100" />
      <el-table-column prop="category" label="类别" width="100" />
      <el-table-column label="标签" width="200">
        <template #default="{ row }">
          <el-tag v-if="row.is_985" type="danger" size="small" style="margin-right: 5px;">985</el-tag>
          <el-tag v-if="row.is_211" type="warning" size="small" style="margin-right: 5px;">211</el-tag>
          <el-tag v-if="row.is_double_first_class" type="success" size="small">双一流</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="batch" label="录取批次" />
      <el-table-column label="操作" width="120">
        <template #default="{ row }">
          <el-button type="primary" size="small" @click="handleDetail(row)">详情</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 高校详情弹窗 -->
    <el-dialog title="高校详情" v-model="dialogVisible" width="50%">
      <div v-if="currentSchool">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="学校名称">{{ currentSchool.name }}</el-descriptions-item>
          <el-descriptions-item label="城市">{{ currentSchool.city }}</el-descriptions-item>
          <el-descriptions-item label="层次">{{ currentSchool.level }}</el-descriptions-item>
          <el-descriptions-item label="类别">{{ currentSchool.category }}</el-descriptions-item>
          <el-descriptions-item label="官方网址"><a :href="currentSchool.website" target="_blank">{{ currentSchool.website }}</a></el-descriptions-item>
          <el-descriptions-item label="招生办电话">{{ currentSchool.phone }}</el-descriptions-item>
          <el-descriptions-item label="简介" :span="2">{{ currentSchool.description }}</el-descriptions-item>
          <el-descriptions-item label="详情" :span="2">{{ currentSchool.details }}</el-descriptions-item>
        </el-descriptions>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.school-container {
  padding: 20px;
}
.header-action {
  margin-bottom: 20px;
  display: flex;
}
</style>
