import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export const useAppStore = defineStore('app', () => {
  // State
  const currentJob = ref(null)
  const stats = ref(null)
  const processes = ref([])
  const tasks = ref([])
  const roles = ref([])
  const decisions = ref([])
  const flows = ref([])
  const bpmnContent = ref('')
  const loading = ref(false)
  const error = ref(null)

  // Actions
  async function uploadFile(file) {
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      loading.value = true
      error.value = null
      const response = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      currentJob.value = {
        ...response.data,
        status: 'pending',
        progress: 0,
        steps: []
      }
      return response.data
    } catch (e) {
      error.value = e.response?.data?.detail || e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  async function startProcessing(jobId) {
    try {
      await axios.post(`/api/process/${jobId}`)
      return true
    } catch (e) {
      error.value = e.response?.data?.detail || e.message
      return false
    }
  }

  function subscribeToJob(jobId, onUpdate) {
    const eventSource = new EventSource(`/api/jobs/${jobId}/stream`)
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      currentJob.value = data
      onUpdate(data)
      
      if (data.status === 'completed' || data.status === 'error') {
        eventSource.close()
      }
    }
    
    eventSource.onerror = () => {
      eventSource.close()
    }
    
    return () => eventSource.close()
  }

  async function fetchStats() {
    try {
      const response = await axios.get('/api/graph-stats')
      stats.value = response.data
    } catch (e) {
      console.error('Failed to fetch stats:', e)
    }
  }

  async function fetchProcesses() {
    try {
      loading.value = true
      error.value = null
      const response = await axios.get('/api/processes')
      console.log('API response:', response.data)
      processes.value = response.data.processes || []
      console.log('Set processes:', processes.value.length)
      if (processes.value.length === 0) {
        console.warn('No processes returned from API')
      }
    } catch (e) {
      console.error('Error fetching processes:', e)
      error.value = e.response?.data?.detail || e.message
      processes.value = []
    } finally {
      loading.value = false
    }
  }

  async function fetchTasks() {
    try {
      loading.value = true
      const response = await axios.get('/api/tasks')
      tasks.value = response.data.tasks
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function fetchRoles() {
    try {
      loading.value = true
      const response = await axios.get('/api/roles')
      roles.value = response.data.roles
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function fetchDecisions() {
    try {
      loading.value = true
      const response = await axios.get('/api/decisions')
      decisions.value = response.data.decisions
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function fetchFlows() {
    try {
      const response = await axios.get('/api/sequence-flows')
      flows.value = response.data.flows
    } catch (e) {
      console.error('Failed to fetch flows:', e)
    }
  }

  async function fetchBpmnContent(processId = null) {
    try {
      const url = processId 
        ? `/api/files/bpmn/content?process_id=${encodeURIComponent(processId)}`
        : '/api/files/bpmn/content'
      const response = await axios.get(url)
      bpmnContent.value = response.data.content
      return response.data.content
    } catch (e) {
      console.error('Failed to fetch BPMN:', e)
      return null
    }
  }

  async function fetchBpmnList() {
    try {
      const response = await axios.get('/api/files/bpmn/list')
      return response.data.files || []
    } catch (e) {
      console.error('Failed to fetch BPMN list:', e)
      return []
    }
  }

  async function fetchAllBpmnContents() {
    try {
      const response = await axios.get('/api/files/bpmn/all')
      return response.data.bpmn_files || {}
    } catch (e) {
      console.error('Failed to fetch all BPMN contents:', e)
      return {}
    }
  }

  async function checkNeo4jStatus() {
    try {
      const response = await axios.get('/api/neo4j/status')
      return response.data
    } catch (e) {
      console.error('Failed to check Neo4j status:', e)
      return { has_data: false, counts: { processes: 0, tasks: 0, roles: 0 } }
    }
  }

  async function clearNeo4j() {
    try {
      await axios.post('/api/neo4j/clear')
      return true
    } catch (e) {
      console.error('Failed to clear Neo4j:', e)
      return false
    }
  }

  return {
    // State
    currentJob,
    stats,
    processes,
    tasks,
    roles,
    decisions,
    flows,
    bpmnContent,
    loading,
    error,
    
    // Actions
    uploadFile,
    startProcessing,
    subscribeToJob,
    fetchStats,
    fetchProcesses,
    fetchTasks,
    fetchRoles,
    fetchDecisions,
    fetchFlows,
    fetchBpmnContent,
    fetchBpmnList,
    fetchAllBpmnContents,
    checkNeo4jStatus,
    clearNeo4j,
  }
})




