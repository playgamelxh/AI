<script setup>
</script>

<template>
  <el-container class="layout-container">
    <el-header>
      <h2>高考志愿辅助系统 (管理后台)</h2>
    </el-header>
    <el-container>
      <el-aside width="200px">
        <el-menu default-active="/schools" router>
          <el-menu-item index="/schools">
            <span>高校管理</span>
          </el-menu-item>
          <el-menu-item index="/majors">
            <span>专业管理</span>
          </el-menu-item>
          <el-menu-item index="/scores">
            <span>历年分数线</span>
          </el-menu-item>
          <el-menu-item index="/provinces">
            <span>省份管理</span>
          </el-menu-item>
          <el-menu-item index="/ai">
            <span>AI智能咨询</span>
          </el-menu-item>
        </el-menu>
      </el-aside>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.layout-container {
  height: 100vh;
}
.el-header {
  background-color: #409eff;
  color: white;
  line-height: 60px;
}
.el-header h2 {
  margin: 0;
}
.el-aside {
  border-right: solid 1px #e6e6e6;
}
</style>
