<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const API_BASE = 'http://localhost:8080/api'
const schools = ref([])
const majors = ref([])
const selectedSchool = ref(null)

const fetchSchools = async () => {
  try {
    const res = await axios.get(`${API_BASE}/schools`)
    schools.value = res.data.data
  } catch (error) {
    ElMessage.error('获取学校列表失败')
  }
}

const fetchMajors = async () => {
  if (!selectedSchool.value) {
    majors.value = []
    return
  }
  try {
    const res = await axios.get(`${API_BASE}/majors`, {
      params: { school_id: selectedSchool.value }
    })
    majors.value = res.data.data
  } catch (error) {
    ElMessage.error('获取专业列表失败')
  }
}

onMounted(() => {
  fetchSchools()
})
</script>

<template>
  <div class="major-container">
    <div class="header-action">
      <el-select v-model="selectedSchool" placeholder="请选择学校" filterable style="width: 300px; margin-right: 10px;" @change="fetchMajors">
        <el-option v-for="school in schools" :key="school.id" :label="school.name" :value="school.id" />
      </el-select>
    </div>
    
    <el-table :data="majors" style="width: 100%" border empty-text="请先选择学校或该学校暂无专业数据">
      <el-table-column prop="id" label="专业ID" width="80" />
      <el-table-column prop="name" label="专业名称" width="200" />
      <el-table-column prop="description" label="专业简介" />
    </el-table>
  </div>
</template>

<style scoped>
.major-container {
  padding: 20px;
}
.header-action {
  margin-bottom: 20px;
}
</style>
