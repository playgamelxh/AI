import { createRouter, createWebHistory } from 'vue-router'
import ProvinceView from '../views/ProvinceView.vue'
import SchoolView from '../views/SchoolView.vue'
import MajorView from '../views/MajorView.vue'
import ScoreView from '../views/ScoreView.vue'
import AIConsultView from '../views/AIConsultView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/schools'
    },
    {
      path: '/provinces',
      name: 'provinces',
      component: ProvinceView
    },
    {
      path: '/schools',
      name: 'schools',
      component: SchoolView
    },
    {
      path: '/majors',
      name: 'majors',
      component: MajorView
    },
    {
      path: '/scores',
      name: 'scores',
      component: ScoreView
    },
    {
      path: '/ai',
      name: 'ai_consult',
      component: AIConsultView
    }
  ]
})

export default router
