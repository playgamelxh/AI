<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

const API_BASE = 'http://localhost:8080/api'
const provinces = ref([])
const dialogVisible = ref(false)
const isEdit = ref(false)
const form = ref({ id: null, code: '', name: '' })

const fetchProvinces = async () => {
  try {
    const res = await axios.get(`${API_BASE}/provinces`)
    provinces.value = res.data.data
  } catch (error) {
    ElMessage.error('获取省份列表失败')
  }
}

onMounted(() => {
  fetchProvinces()
})

const handleAdd = () => {
  isEdit.value = false
  form.value = { id: null, code: '', name: '' }
  dialogVisible.value = true
}

const handleEdit = (row) => {
  isEdit.value = true
  form.value = { ...row }
  dialogVisible.value = true
}

const handleDelete = async (id) => {
  try {
    await ElMessageBox.confirm('确认删除该省份吗？', '提示', { type: 'warning' })
    await axios.delete(`${API_BASE}/provinces/${id}`)
    ElMessage.success('删除成功')
    fetchProvinces()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error('删除失败')
  }
}

const handleSave = async () => {
  try {
    if (isEdit.value) {
      await axios.put(`${API_BASE}/provinces/${form.value.id}`, form.value)
      ElMessage.success('更新成功')
    } else {
      await axios.post(`${API_BASE}/provinces`, form.value)
      ElMessage.success('添加成功')
    }
    dialogVisible.value = false
    fetchProvinces()
  } catch (error) {
    ElMessage.error('保存失败')
  }
}
</script>

<template>
  <div class="province-container">
    <div class="header-action">
      <el-button type="primary" @click="handleAdd">新增省份</el-button>
    </div>
    
    <el-table :data="provinces" style="width: 100%" border>
      <el-table-column prop="id" label="ID" width="80" />
      <el-table-column prop="code" label="省份编号" />
      <el-table-column prop="name" label="省份名称" />
      <el-table-column label="操作" width="180">
        <template #default="{ row }">
          <el-button size="small" @click="handleEdit(row)">编辑</el-button>
          <el-button size="small" type="danger" @click="handleDelete(row.id)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog :title="isEdit ? '编辑省份' : '新增省份'" v-model="dialogVisible" width="30%">
      <el-form :model="form" label-width="80px">
        <el-form-item label="省份编号">
          <el-input v-model="form.code" />
        </el-form-item>
        <el-form-item label="省份名称">
          <el-input v-model="form.name" />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" @click="handleSave">确认</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.province-container {
  padding: 20px;
}
.header-action {
  margin-bottom: 20px;
}
</style>
