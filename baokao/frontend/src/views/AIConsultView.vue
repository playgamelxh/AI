<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const API_BASE = 'http://localhost:8080/api'
const form = ref({
  score: null,
  province: '',
  type: '物理类',
  city_pref: '',
  major_pref: '',
  interests: '',
  subjects: ''
})

const loading = ref(false)
const result = ref(null)

const handleConsult = async () => {
  if (!form.value.score || !form.value.province) {
    ElMessage.warning('请填写高考分数和省份')
    return
  }
  
  loading.value = true
  try {
    const res = await axios.post(`${API_BASE}/ai/consult`, form.value)
    result.value = res.data.data
    ElMessage.success('咨询结果生成成功')
  } catch (error) {
    ElMessage.error('AI服务请求失败，请稍后重试')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="ai-container">
    <el-card class="consult-form">
      <template #header>
        <div class="card-header">
          <span>AI 智能志愿填报咨询</span>
        </div>
      </template>
      <el-form :model="form" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="高考分数" required>
              <el-input-number v-model="form.score" :min="0" :max="750" placeholder="请输入分数" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="所在省份" required>
              <el-input v-model="form.province" placeholder="例如：北京、广东、浙江" />
            </el-form-item>
          </el-col>
        </el-row>
        
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="报考科类" required>
              <el-radio-group v-model="form.type">
                <el-radio label="物理类">物理类 / 理科</el-radio>
                <el-radio label="历史类">历史类 / 文科</el-radio>
                <el-radio label="综合">综合</el-radio>
              </el-radio-group>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="意向城市">
              <el-input v-model="form.city_pref" placeholder="例如：北京, 上海, 广州 (选填)" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item label="意向专业">
          <el-input v-model="form.major_pref" placeholder="例如：计算机, 医学, 师范 (选填)" />
        </el-form-item>
        
        <el-form-item label="兴趣爱好">
          <el-input type="textarea" v-model="form.interests" placeholder="简述你的兴趣爱好，以便AI更好地推荐 (选填)" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleConsult" :loading="loading">生成报考方案</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card v-if="result" class="consult-result" style="margin-top: 20px;">
      <template #header>
        <div class="card-header">
          <span>AI 推荐结果</span>
        </div>
      </template>
      <div class="result-content">
        <p class="summary">{{ result.summary }}</p>
        
        <div class="batch-section">
          <h3>冲刺批次 (建议把以下院校放在志愿前列)</h3>
          <ul>
            <li v-for="(item, index) in result.batch_a" :key="'a'+index">{{ item }}</li>
          </ul>
        </div>

        <div class="batch-section">
          <h3>稳妥保底批次 (平行志愿推荐)</h3>
          <ul>
            <li v-for="(item, index) in result.parallel" :key="'p'+index">{{ item }}</li>
          </ul>
        </div>

        <div class="suggestion-section">
          <h3>AI 报考建议</h3>
          <p>{{ result.suggestions }}</p>
        </div>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.ai-container {
  padding: 20px;
  max-width: 900px;
  margin: 0 auto;
}
.card-header {
  font-weight: bold;
  font-size: 18px;
}
.result-content {
  line-height: 1.6;
}
.summary {
  font-size: 16px;
  color: #409EFF;
  margin-bottom: 20px;
}
.batch-section h3, .suggestion-section h3 {
  font-size: 16px;
  color: #303133;
  border-left: 4px solid #409EFF;
  padding-left: 10px;
  margin-top: 20px;
}
ul {
  padding-left: 20px;
}
li {
  margin-bottom: 8px;
  color: #606266;
}
.suggestion-section p {
  color: #606266;
  background-color: #f4f4f5;
  padding: 15px;
  border-radius: 4px;
}
</style>
