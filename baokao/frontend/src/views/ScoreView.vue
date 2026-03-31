<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const API_BASE = 'http://localhost:8080/api'
const schools = ref([])
const provinces = ref([])
const scores = ref([])

const filterSchool = ref(null)
const filterProvince = ref(null)
const filterYear = ref(2025)

const fetchOptions = async () => {
  try {
    const [resSchools, resProvinces] = await Promise.all([
      axios.get(`${API_BASE}/schools`),
      axios.get(`${API_BASE}/provinces`)
    ])
    schools.value = resSchools.data.data
    provinces.value = resProvinces.data.data
  } catch (error) {
    ElMessage.error('获取基础数据失败')
  }
}

const fetchScores = async () => {
  if (!filterSchool.value) {
    ElMessage.warning('请选择查询学校')
    return
  }
  try {
    const res = await axios.get(`${API_BASE}/scores`, {
      params: { 
        school_id: filterSchool.value,
        province_id: filterProvince.value || '',
        year: filterYear.value || ''
      }
    })
    scores.value = res.data.data
  } catch (error) {
    ElMessage.error('获取分数线失败')
  }
}

onMounted(() => {
  fetchOptions()
})
</script>

<template>
  <div class="score-container">
    <div class="header-action">
      <el-select v-model="filterSchool" placeholder="选择学校" filterable style="width: 200px; margin-right: 10px;">
        <el-option v-for="s in schools" :key="s.id" :label="s.name" :value="s.id" />
      </el-select>
      <el-select v-model="filterProvince" placeholder="招生省份(选填)" filterable clearable style="width: 150px; margin-right: 10px;">
        <el-option v-for="p in provinces" :key="p.id" :label="p.name" :value="p.id" />
      </el-select>
      <el-select v-model="filterYear" placeholder="年份" clearable style="width: 120px; margin-right: 10px;">
        <el-option label="2025" :value="2025" />
        <el-option label="2024" :value="2024" />
      </el-select>
      <el-button type="primary" @click="fetchScores">查询分数线</el-button>
    </div>
    
    <el-table :data="scores" style="width: 100%" border empty-text="暂无分数线数据">
      <el-table-column prop="year" label="年份" width="100" />
      <el-table-column prop="major_id" label="专业ID" width="100" />
      <el-table-column prop="province_id" label="招生省份ID" width="100" />
      <el-table-column prop="type" label="科类" width="100" />
      <el-table-column prop="batch" label="录取批次" width="120" />
      <el-table-column prop="lowest_score" label="最低分数线" width="120" />
      <el-table-column prop="admission_count" label="录取人数" />
    </el-table>
  </div>
</template>

<style scoped>
.score-container {
  padding: 20px;
}
.header-action {
  margin-bottom: 20px;
  display: flex;
}
</style>
