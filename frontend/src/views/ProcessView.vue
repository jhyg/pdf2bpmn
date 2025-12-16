<template>
  <div class="h-[calc(100vh-8rem)] flex flex-col">
    <!-- Header -->
    <div class="glass border-b border-white/10 px-6 py-4">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold">프로세스 뷰어</h1>
          <p class="text-gray-400 text-sm">BPMN.io로 시각화된 프로세스 다이어그램</p>
        </div>
        <div class="flex items-center space-x-3">
          <button 
            @click="zoomIn"
            class="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-all"
            title="확대"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
          </button>
          <button 
            @click="zoomOut"
            class="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-all"
            title="축소"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
            </svg>
          </button>
          <button 
            @click="fitToViewport"
            class="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-all"
            title="화면에 맞추기"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
          <a 
            href="/api/files/bpmn"
            download="process.bpmn"
            class="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-sm font-medium hover:from-blue-600 hover:to-purple-700 transition-all"
          >
            📥 BPMN 다운로드
          </a>
        </div>
      </div>
    </div>

    <!-- BPMN Viewer -->
    <div class="flex-1 flex">
      <!-- Diagram Area -->
      <div class="flex-1 relative bg-slate-900/50">
        <div v-if="loading" class="absolute inset-0 flex items-center justify-center">
          <div class="text-center">
            <svg class="w-12 h-12 animate-spin text-blue-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p class="text-gray-400">BPMN 다이어그램 로딩 중...</p>
          </div>
        </div>
        
        <div v-if="error" class="absolute inset-0 flex items-center justify-center">
          <div class="text-center">
            <svg class="w-12 h-12 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <p class="text-gray-400">{{ error }}</p>
            <button @click="loadBpmn" class="mt-4 px-4 py-2 bg-blue-500 rounded-lg">다시 시도</button>
          </div>
        </div>
        
        <div ref="bpmnContainer" class="w-full h-full"></div>
      </div>

      <!-- Side Panel: Selected Element Info -->
      <div v-if="selectedElement" class="w-96 glass border-l border-white/10 overflow-y-auto">
        <div class="p-4">
          <div class="flex items-center justify-between mb-4">
            <h3 class="font-semibold">선택된 요소</h3>
            <button @click="selectedElement = null" class="text-gray-400 hover:text-white">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <!-- Loading State -->
          <div v-if="loadingDetail" class="flex items-center justify-center py-8">
            <svg class="w-6 h-6 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
          </div>
          
          <!-- Element Info -->
          <div v-else class="space-y-4">
            <!-- Basic Info -->
            <div class="p-3 rounded-lg bg-white/5">
              <div class="flex items-center space-x-2 mb-2">
                <span class="px-2 py-0.5 text-xs rounded-full" 
                      :class="elementTypeClass(selectedElement.type)">
                  {{ selectedElement.type }}
                </span>
              </div>
              <h4 class="text-lg font-semibold">{{ selectedElement.name || selectedElement.id }}</h4>
            </div>
            
            <!-- Sequence Flow Info (when clicking on a flow/arrow) -->
            <div v-if="selectedElement.isSequenceFlow" class="space-y-3">
              <!-- Flow Direction -->
              <div class="p-3 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
                <label class="text-xs text-cyan-400 uppercase tracking-wider flex items-center space-x-1">
                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                  <span>흐름 방향</span>
                </label>
                <div class="mt-2 flex items-center space-x-2">
                  <span class="text-sm text-gray-300">{{ selectedElement.sourceRef }}</span>
                  <svg class="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                  <span class="text-sm text-gray-300">{{ selectedElement.targetRef }}</span>
                </div>
              </div>
              
              <!-- Condition (if exists) -->
              <div v-if="selectedElement.condition" class="p-4 rounded-lg bg-yellow-500/20 border-2 border-yellow-500/50">
                <label class="text-xs text-yellow-400 uppercase tracking-wider flex items-center space-x-1 mb-2">
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>조건 (Condition)</span>
                </label>
                <p class="text-base font-medium text-yellow-200">{{ selectedElement.condition }}</p>
              </div>
              
              <!-- No Condition -->
              <div v-else class="p-3 rounded-lg bg-gray-500/10 border border-gray-500/30">
                <p class="text-sm text-gray-400 text-center">
                  조건 없음 (기본 흐름)
                </p>
              </div>
            </div>
            
            <!-- Description (for non-sequence-flow elements) -->
            <div v-if="!selectedElement.isSequenceFlow && elementDetail?.element?.description" class="p-3 rounded-lg bg-blue-500/10 border border-blue-500/30">
              <label class="text-xs text-blue-400 uppercase tracking-wider flex items-center space-x-1">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>설명</span>
              </label>
              <p class="text-sm text-gray-300 mt-2">{{ elementDetail.element.description }}</p>
            </div>
            
            <!-- Role Info (not for sequence flows) -->
            <div v-if="!selectedElement.isSequenceFlow && elementDetail?.role" class="p-3 rounded-lg bg-purple-500/10 border border-purple-500/30">
              <label class="text-xs text-purple-400 uppercase tracking-wider flex items-center space-x-1">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                <span>담당자</span>
              </label>
              <p class="font-medium mt-2">{{ elementDetail.role.name }}</p>
              <p v-if="elementDetail.role.description" class="text-xs text-gray-400 mt-1">
                {{ elementDetail.role.description }}
              </p>
            </div>
            
            <!-- Process Info (not for sequence flows) -->
            <div v-if="!selectedElement.isSequenceFlow && elementDetail?.process" class="p-3 rounded-lg bg-green-500/10 border border-green-500/30">
              <label class="text-xs text-green-400 uppercase tracking-wider flex items-center space-x-1">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6z" />
                </svg>
                <span>소속 프로세스</span>
              </label>
              <p class="font-medium mt-2">{{ elementDetail.process.name }}</p>
            </div>
            
            <!-- Sequence Info (not for sequence flows) -->
            <div v-if="!selectedElement.isSequenceFlow && (elementDetail?.next_tasks?.length || elementDetail?.prev_tasks?.length)" 
                 class="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
              <label class="text-xs text-yellow-400 uppercase tracking-wider flex items-center space-x-1">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
                <span>흐름</span>
              </label>
              <div class="mt-2 space-y-2">
                <div v-if="elementDetail?.prev_tasks?.length" class="flex items-center text-sm">
                  <span class="text-gray-400 mr-2">이전:</span>
                  <span v-for="(task, i) in elementDetail.prev_tasks" :key="task.task_id" class="text-gray-300">
                    {{ task.name }}<span v-if="i < elementDetail.prev_tasks.length - 1">, </span>
                  </span>
                </div>
                <div v-if="elementDetail?.next_tasks?.length" class="flex items-center text-sm">
                  <span class="text-gray-400 mr-2">다음:</span>
                  <span v-for="(task, i) in elementDetail.next_tasks" :key="task.task_id" class="text-gray-300">
                    {{ task.name }}<span v-if="i < elementDetail.next_tasks.length - 1">, </span>
                  </span>
                </div>
              </div>
            </div>
            
            <!-- Evidence / Source Documents (not for sequence flows) -->
            <div v-if="!selectedElement.isSequenceFlow && elementDetail?.evidences?.length" class="space-y-3">
              <label class="text-xs text-orange-400 uppercase tracking-wider flex items-center space-x-1">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>출처 문서</span>
              </label>
              
              <div v-for="(evidence, idx) in elementDetail.evidences" :key="idx" 
                   class="p-3 rounded-lg bg-orange-500/10 border border-orange-500/30">
                <div class="flex items-center justify-between mb-2">
                  <span class="text-xs font-mono text-orange-400">
                    📄 페이지 {{ evidence.page || '?' }}
                  </span>
                </div>
                <p class="text-sm text-gray-300 leading-relaxed">
                  "{{ truncateText(evidence.text, 300) }}"
                </p>
              </div>
            </div>
            
            <!-- No Evidence (not for sequence flows) -->
            <div v-else-if="!selectedElement.isSequenceFlow && !loadingDetail" class="p-3 rounded-lg bg-gray-500/10 border border-gray-500/30">
              <p class="text-sm text-gray-400 text-center">
                출처 문서 정보가 없습니다
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useAppStore } from '../stores/app'
import axios from 'axios'

const store = useAppStore()
const bpmnContainer = ref(null)
const loading = ref(true)
const error = ref(null)
const selectedElement = ref(null)
const elementDetail = ref(null)
const loadingDetail = ref(false)

let bpmnViewer = null

onMounted(async () => {
  await loadBpmn()
})

onUnmounted(() => {
  if (bpmnViewer) {
    bpmnViewer.destroy()
  }
})

// Watch for element selection and fetch details
watch(selectedElement, async (newElement) => {
  if (newElement) {
    await fetchElementDetail(newElement)
  } else {
    elementDetail.value = null
  }
})

async function fetchElementDetail(element) {
  loadingDetail.value = true
  elementDetail.value = null
  
  try {
    // Try to fetch by element name (more reliable than BPMN ID)
    const response = await axios.get(`/api/bpmn/element/${encodeURIComponent(element.name || element.id)}`)
    
    if (response.data.found) {
      elementDetail.value = response.data
    } else {
      // If not found by name, try a more flexible search
      const searchTerm = element.name || element.id.replace(/Activity_|Gateway_|Event_|StartEvent_|EndEvent_/g, '').replace(/_/g, ' ')
      const retryResponse = await axios.get(`/api/bpmn/element/${encodeURIComponent(searchTerm)}`)
      
      if (retryResponse.data.found) {
        elementDetail.value = retryResponse.data
      }
    }
  } catch (e) {
    console.error('Failed to fetch element detail:', e)
  } finally {
    loadingDetail.value = false
  }
}

function truncateText(text, maxLength = 200) {
  if (!text) return ''
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

function elementTypeClass(type) {
  const classes = {
    'Task': 'bg-blue-500/20 text-blue-400',
    'UserTask': 'bg-blue-500/20 text-blue-400',
    'ServiceTask': 'bg-cyan-500/20 text-cyan-400',
    'Gateway': 'bg-yellow-500/20 text-yellow-400',
    'ExclusiveGateway': 'bg-yellow-500/20 text-yellow-400',
    'ParallelGateway': 'bg-orange-500/20 text-orange-400',
    'StartEvent': 'bg-green-500/20 text-green-400',
    'EndEvent': 'bg-red-500/20 text-red-400',
    'Event': 'bg-purple-500/20 text-purple-400',
    'SequenceFlow': 'bg-cyan-500/20 text-cyan-400',
  }
  return classes[type] || 'bg-gray-500/20 text-gray-400'
}

async function loadBpmn() {
  loading.value = true
  error.value = null
  
  try {
    const content = await store.fetchBpmnContent()
    
    if (!content) {
      error.value = 'BPMN 파일을 찾을 수 없습니다. 먼저 PDF를 변환해주세요.'
      loading.value = false
      return
    }
    
    // Dynamically import bpmn-js
    const { default: BpmnViewer } = await import('bpmn-js/lib/NavigatedViewer')
    
    if (bpmnViewer) {
      bpmnViewer.destroy()
    }
    
    bpmnViewer = new BpmnViewer({
      container: bpmnContainer.value,
      keyboard: { bindTo: document }
    })
    
    // Apply dark theme styling
    const canvas = bpmnViewer.get('canvas')
    
    await bpmnViewer.importXML(content)
    
    // Fit to viewport
    canvas.zoom('fit-viewport')
    
    // Add element selection handler
    const eventBus = bpmnViewer.get('eventBus')
    eventBus.on('element.click', (e) => {
      const element = e.element
      if (element.type !== 'bpmn:Process' && element.type !== 'label') {
        const bo = element.businessObject
        
        // Handle SequenceFlow with condition
        if (element.type === 'bpmn:SequenceFlow') {
          const condition = bo?.conditionExpression?.body || bo?.name || null
          const sourceRef = bo?.sourceRef?.name || bo?.sourceRef?.id || 'Unknown'
          const targetRef = bo?.targetRef?.name || bo?.targetRef?.id || 'Unknown'
          
          selectedElement.value = {
            id: element.id,
            name: bo?.name || `${sourceRef} → ${targetRef}`,
            type: 'SequenceFlow',
            description: condition ? `조건: ${condition}` : null,
            isSequenceFlow: true,
            condition: condition,
            sourceRef: sourceRef,
            targetRef: targetRef
          }
        } else {
          selectedElement.value = {
            id: element.id,
            name: bo?.name || element.id,
            type: element.type.replace('bpmn:', ''),
            description: bo?.documentation?.[0]?.text,
            isSequenceFlow: false
          }
        }
      }
    })
    
    loading.value = false
    
  } catch (e) {
    console.error('Failed to load BPMN:', e)
    error.value = 'BPMN 로딩 실패: ' + e.message
    loading.value = false
  }
}

function zoomIn() {
  if (bpmnViewer) {
    const canvas = bpmnViewer.get('canvas')
    canvas.zoom(canvas.zoom() * 1.2)
  }
}

function zoomOut() {
  if (bpmnViewer) {
    const canvas = bpmnViewer.get('canvas')
    canvas.zoom(canvas.zoom() * 0.8)
  }
}

function fitToViewport() {
  if (bpmnViewer) {
    const canvas = bpmnViewer.get('canvas')
    canvas.zoom('fit-viewport')
  }
}
</script>

<style>
.bjs-powered-by {
  display: none !important;
}

/* BPMN Elements (Tasks, Events, Gateways) */
.djs-element .djs-visual > :first-child {
  fill: #1e293b !important;
  stroke: #94a3b8 !important;
}

.djs-element .djs-visual text {
  fill: #f1f5f9 !important;
}

/* Sequence Flows (Lines/Arrows) - Bright color */
.djs-connection .djs-visual path {
  stroke: #38bdf8 !important;
  stroke-width: 2px !important;
}

.djs-connection .djs-visual polyline {
  stroke: #38bdf8 !important;
  stroke-width: 2px !important;
}

/* All SVG paths in connections */
g.djs-group g.djs-element.djs-connection g.djs-visual path {
  stroke: #38bdf8 !important;
  stroke-width: 2px !important;
}

/* Sequence Flow Arrow markers (defined in defs) */
svg defs marker path {
  fill: #38bdf8 !important;
  stroke: #38bdf8 !important;
}

/* Sequence Flow Labels */
.djs-connection .djs-visual text,
.djs-label text {
  fill: #fbbf24 !important;
  font-size: 11px !important;
}

/* Hover effects */
.djs-element:hover .djs-visual > :first-child {
  stroke: #22d3ee !important;
}

.djs-connection:hover .djs-visual path,
.djs-connection:hover .djs-visual polyline {
  stroke: #67e8f9 !important;
  stroke-width: 3px !important;
}

/* Selected element */
.djs-element.selected .djs-visual > :first-child {
  stroke: #a78bfa !important;
  stroke-width: 2px !important;
}

.djs-connection.selected .djs-visual path,
.djs-connection.selected .djs-visual polyline {
  stroke: #c4b5fd !important;
  stroke-width: 3px !important;
}
</style>




