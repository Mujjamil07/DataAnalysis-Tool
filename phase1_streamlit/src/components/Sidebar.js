import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  LayoutDashboard, 
  Upload, 
  Brain, 
  BarChart3,
  Database,
  Settings,
  HelpCircle,
  LogOut,
  Scissors
} from 'lucide-react';
import { useAppStore } from '../store/appStore';

const Sidebar = ({ isOpen, onClose, data, models, results }) => {
  const location = useLocation();
  const { getDataStats, getModelStats } = useAppStore();
  const dataStats = getDataStats();
  const modelStats = getModelStats();

  const menuItems = [
    {
      path: '/',
      icon: LayoutDashboard,
      label: 'Dashboard',
      description: 'Overview & Analytics'
    },
    {
      path: '/upload',
      icon: Upload,
      label: 'Data Upload',
      description: 'Upload & Analyze Data',
      badge: data ? '✓' : null
    },
    {
      path: '/cleaning',
      icon: Scissors,
      label: 'Data Cleaning',
      description: 'Clean & Prepare Data',
      badge: data ? '🧹' : null
    },
    {
      path: '/visualizations',
      icon: BarChart3,
      label: 'Visualizations',
      description: 'Data Analysis & Charts',
      badge: data ? '📊' : null
    },
    {
      path: '/training',
      icon: Brain,
      label: 'Model Training',
      description: 'Train AI Models',
      badge: null  // Always show without badge
    },
    {
      path: '/results',
      icon: BarChart3,
      label: 'Model Results',
      description: 'View Performance',
      badge: Object.keys(results).length > 0 ? Object.keys(results).length : null
    },
    {
      path: '/enhanced-analysis',
      icon: Brain,
      label: 'Enhanced Analysis',
      description: 'Advanced ML Pipeline',
      badge: '🚀'
    }
  ];

  const sidebarVariants = {
    closed: {
      x: '-100%',
      transition: {
        type: 'tween',
        duration: 0.3
      }
    },
    open: {
      x: 0,
      transition: {
        type: 'tween',
        duration: 0.3
      }
    }
  };

  const renderSidebarContent = () => (
    <>
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-watson-700 to-watson-800 rounded-lg flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-gradient font-ibm">
              Watson AI Studio
            </h2>
            <p className="text-xs text-gray-500 font-inter">
              AutoML Platform
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="lg:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <X className="w-5 h-5 text-gray-600" />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-6 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              onClick={onClose}
              className={`sidebar-item ${isActive ? 'active' : ''}`}
            >
              <Icon className="w-5 h-5 mr-3" />
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{item.label}</span>
                  {item.badge && (
                    <span className="px-2 py-1 text-xs bg-watson-100 text-watson-700 rounded-full">
                      {item.badge}
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-1">{item.description}</p>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Stats Section */}
      {(dataStats || modelStats.totalModels > 0) && (
        <div className="p-6 border-t border-gray-200">
          <h3 className="text-sm font-semibold text-gray-900 mb-3">Quick Stats</h3>
          <div className="space-y-2">
            {dataStats && dataStats.rows && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Dataset Size</span>
                <span className="font-medium">{dataStats.rows.toLocaleString()}</span>
              </div>
            )}
            {dataStats && dataStats.columns && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Features</span>
                <span className="font-medium">{dataStats.columns}</span>
              </div>
            )}
            {modelStats.totalModels > 0 && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Models Trained</span>
                <span className="font-medium">{modelStats.trainedModels}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="p-6 border-t border-gray-200 space-y-2">
        <button className="sidebar-item w-full">
          <Settings className="w-5 h-5 mr-3" />
          <span>Settings</span>
        </button>
        <button className="sidebar-item w-full">
          <HelpCircle className="w-5 h-5 mr-3" />
          <span>Help & Support</span>
        </button>
        <button className="sidebar-item w-full text-red-600 hover:text-red-700">
          <LogOut className="w-5 h-5 mr-3" />
          <span>Logout</span>
        </button>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* Mobile Sidebar */}
      <motion.aside
        variants={sidebarVariants}
        initial="closed"
        animate={isOpen ? "open" : "closed"}
        className="fixed lg:hidden inset-y-0 left-0 z-50 w-80 bg-white border-r border-gray-200 flex flex-col"
      >
        {renderSidebarContent()}
      </motion.aside>

      {/* Desktop Sidebar - Always Visible */}
      <aside className="hidden lg:flex lg:relative lg:inset-y-0 lg:left-0 lg:z-50 lg:w-80 lg:bg-white lg:border-r lg:border-gray-200 lg:flex-col">
        {renderSidebarContent()}
      </aside>
    </>
  );
};

export default Sidebar;
