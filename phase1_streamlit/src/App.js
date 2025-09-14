import React, { useState } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import DataUpload from './pages/DataUpload';
import ModelTraining from './pages/ModelTraining_fresh';
import ModelResults from './pages/ModelResults';
import Visualizations from './pages/Visualizations';
import DataCleaning from './pages/DataCleaning';
import EnhancedAnalysis from './pages/EnhancedAnalysis';
import { useAppStore } from './store/appStore';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const { data, models, results } = useAppStore();

  const pageVariants = {
    initial: {
      opacity: 0,
      x: -20
    },
    in: {
      opacity: 1,
      x: 0
    },
    out: {
      opacity: 0,
      x: 20
    }
  };

  const pageTransition = {
    type: "tween",
    ease: "anticipate",
    duration: 0.3
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)}
        data={data}
        models={models}
        results={results}
      />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden lg:ml-0">
        {/* Navbar */}
        <Navbar 
          onMenuClick={() => setSidebarOpen(true)}
          data={data}
          models={models}
        />
        
        {/* Page Content */}
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={pageTransition}
              className="h-full"
            >
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/upload" element={<DataUpload />} />
                <Route path="/cleaning" element={<DataCleaning />} />
                <Route path="/visualizations" element={<Visualizations />} />
                <Route path="/training" element={<ModelTraining />} />
                <Route path="/results" element={<ModelResults />} />
                <Route path="/enhanced-analysis" element={<EnhancedAnalysis />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
