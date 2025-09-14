import { create } from 'zustand';

// Simple data persistence without localStorage
const getStoredData = () => {
  try {
    const stored = sessionStorage.getItem('automl-data');
    return stored ? JSON.parse(stored) : null;
  } catch (error) {
    console.warn('Failed to get stored data:', error);
    return null;
  }
};

const setStoredData = (data) => {
  try {
    if (data) {
      sessionStorage.setItem('automl-data', JSON.stringify(data));
    } else {
      sessionStorage.removeItem('automl-data');
    }
  } catch (error) {
    console.warn('Failed to store data:', error);
  }
};

export const useAppStore = create((set, get) => ({
  // Data state - initialize with stored data
  data: getStoredData(),
  targetColumn: sessionStorage.getItem('automl-target-column') || null,
  cleanedData: null,
  
  // Backend session state
  sessionId: null,
  
  // Visualization state
  visualizations: {},
  dataQuality: null,
  
  // Model state
  models: [],
  bestModel: null,
  trainingStatus: 'idle', // idle, training, completed, error
  
  // Results state
  results: {},
  
  // UI state
  currentPage: 'dashboard',
  sidebarOpen: false,
  
  // Actions
  setData: (data) => {
    try {
      console.log('Setting data:', data);
      
      // If data is being restored from storage and has no rawData, mark it as restored
      if (data && !data.rawData && data.rows) {
        const restoredData = {
          ...data,
          restoredFromStorage: true,
          fullDataAvailable: false // Indicate that full data is not available
        };
        set({ data: restoredData });
        setStoredData(restoredData);
        return;
      }
      
      // Check if this is new data (not restored from storage)
      const currentData = get().data;
      const isNewData = !currentData || 
        (data && currentData && (
          data.rows !== currentData.rows || 
          data.columns?.length !== currentData.columns?.length ||
          data.filename !== currentData.filename ||
          data.name !== currentData.name ||
          JSON.stringify(data.columns) !== JSON.stringify(currentData.columns)
        ));
      
      // If new data is uploaded, clear previous model results FIRST
      if (isNewData && data) {
        console.log('New dataset detected, clearing previous model results');
        set({ 
          models: [],
          results: {},
          bestModel: null,
          trainingStatus: 'idle',
          sessionId: null
        });
      }
      
      // Also clear models if this is fresh data (not restored from storage)
      if (data && !data.restoredFromStorage) {
        console.log('Fresh dataset detected, clearing previous model results');
        set({ 
          models: [],
          results: {},
          bestModel: null,
          trainingStatus: 'idle',
          sessionId: null
        });
      }
      
      // Prepare the data to be set - FULL DATASET, NO TRUNCATION, NO LIMITS
      let dataToSet = data;
      // Keep ALL data - no truncation, no limits
      dataToSet = {
        ...data,
        fullDataAvailable: true, // Flag to indicate we have full data
        rawData: data.rawData || data.preview, // Keep ALL raw data
        rows: data.rows, // Keep original row count
        columns: data.columns // Keep all columns
      };
      
      // Set the data
      set({ data: dataToSet });
      setStoredData(dataToSet);
      console.log('Data set successfully:', dataToSet);
      
    } catch (error) {
      console.warn('Failed to set data:', error);
      // Don't use localStorage, just set data directly
      set({ data });
      setStoredData(data);
    }
  },

  setTargetColumn: (column) => {
    console.log('Setting target column:', column);
    set({ targetColumn: column });
    // Store target column in sessionStorage
    try {
      if (column) {
        sessionStorage.setItem('automl-target-column', column);
      } else {
        sessionStorage.removeItem('automl-target-column');
      }
    } catch (error) {
      console.warn('Failed to store target column:', error);
    }
  },
  
  setCleanedData: (cleanedData) => set({ cleanedData }),
  
  setSessionId: (sessionId) => set({ sessionId }),
  
  // Model actions
  setModels: (models) => set({ models }),
  
  setBestModel: (model) => set({ bestModel: model }),
  
  setTrainingStatus: (status) => set({ trainingStatus: status }),
  
  // Results actions
  setResults: (results) => set({ results }),
  
  // UI actions
  setCurrentPage: (page) => set({ currentPage: page }),
  
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  
  // Complex actions
  addModel: (model) => {
    const { models } = get();
    set({ models: [...models, model] });
  },
  
  updateResults: (modelId, results) => {
    const { results: currentResults } = get();
    set({ results: { ...currentResults, [modelId]: results } });
  },
  
  clearData: () => {
    set({ 
      data: null, 
      targetColumn: null, 
      cleanedData: null,
      models: [],
      bestModel: null,
      results: {},
      trainingStatus: 'idle',
      sessionId: null
    });
    // Clear stored data
    setStoredData(null);
    try {
      sessionStorage.removeItem('automl-target-column');
    } catch (error) {
      console.warn('Failed to clear stored target column:', error);
    }
  },
  
  clearModels: () => {
    console.log('Manually clearing all model results');
    set({
      models: [],
      results: {},
      bestModel: null,
      trainingStatus: 'idle',
      sessionId: null
    });
  },
  
  // Visualization actions
  setVisualizations: (visualizations) => {
    set({ visualizations });
  },
  
  setDataQuality: (dataQuality) => {
    set({ dataQuality });
  },
  
  // Computed values
  getDataStats: () => {
    const { data } = get();
    if (!data) return {
      rows: 0,
      columns: 0,
      memoryUsage: 0,
      hasMissingValues: false,
      dataTypes: {}
    };
    
    return {
      rows: data.length || 0,
      columns: data.columns?.length || 0,
      memoryUsage: JSON.stringify(data).length || 0,
      hasMissingValues: data.isnull?.sum().sum() > 0 || false,
      dataTypes: data.dtypes?.to_dict() || {}
    };
  },
  
  getModelStats: () => {
    const { models, results } = get();
    
    // Ensure models is an array
    if (!Array.isArray(models)) {
      return {
        totalModels: 0,
        trainedModels: 0,
        bestModel: null,
        averageAccuracy: 0
      };
    }
    
    const modelNames = models.map(m => m.id);
    
    return {
      totalModels: modelNames.length,
      trainedModels: models.filter(m => m.status === 'completed').length,
      bestModel: get().bestModel,
      averageAccuracy: modelNames.length > 0 
        ? modelNames.reduce((acc, name) => {
            const modelResult = results[name];
            return acc + (modelResult?.accuracy || 0);
          }, 0) / modelNames.length
        : 0
    };
  }
}));

// Initialize target column from sessionStorage
const storedTargetColumn = sessionStorage.getItem('automl-target-column');
if (storedTargetColumn) {
  useAppStore.getState().setTargetColumn(storedTargetColumn);
}
