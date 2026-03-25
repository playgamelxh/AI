<template>
  <div class="app">
    <el-container>
      <el-header>
        <h1 class="title">AI Agent With Skill</h1>
      </el-header>
      <el-main>
        <el-tabs v-model="activeTab">
          <el-tab-pane label="大模型配置" name="llm">
            <div class="placeholder">大模型配置管理 - 开发中...</div>
          </el-tab-pane>
          <el-tab-pane label="AI 对话" name="chat">
            <div class="placeholder">AI 对话功能 - 开发中...</div>
          </el-tab-pane>
          <el-tab-pane label="技能管理" name="skills">
            <div class="placeholder">技能管理 - 开发中...</div>
          </el-tab-pane>
        </el-tabs>
      </el-main>
    </el-container>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const activeTab = ref('llm')
</script>

<style scoped>
.app {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.el-header {
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  padding: 0 40px;
}

.title {
  margin: 0;
  font-size: 24px;
  font-weight: 500;
  color: #303133;
}

.el-main {
  padding: 30px 40px;
}

.placeholder {
  text-align: center;
  padding: 100px 0;
  color: #909399;
  font-size: 16px;
}
</style>
