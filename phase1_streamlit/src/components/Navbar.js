import React from 'react';
import { motion } from 'framer-motion';
import { 
  Menu, 
  Brain, 
  Database, 
  TrendingUp, 
  Settings,
  Bell,
  User
} from 'lucide-react';
import { useAppStore } from '../store/appStore';

const Navbar = ({ onMenuClick, data, models }) => {
  const { getDataStats, getModelStats } = useAppStore();
  const dataStats = getDataStats();
  const modelStats = getModelStats();

  return (
    <motion.nav 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm"
    >
      <div className="flex items-center justify-between">
        {/* Left side */}
        <div className="flex items-center space-x-4">
          <button
            onClick={onMenuClick}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors lg:hidden"
          >
            <Menu className="w-5 h-5 text-gray-600" />
          </button>
          
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-watson-700 to-watson-800 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gradient font-ibm">
                Watson AI Studio
              </h1>
              <p className="text-sm text-gray-500 font-inter">
                Professional AutoML Platform
              </p>
            </div>
          </div>
        </div>

        {/* Center - Quick Stats */}
        <div className="hidden md:flex items-center space-x-6">
          {dataStats && dataStats.rows && (
            <div className="flex items-center space-x-2 text-sm">
              <Database className="w-4 h-4 text-watson-600" />
              <span className="text-gray-600">
                {dataStats.rows.toLocaleString()} rows
              </span>
            </div>
          )}
          
          {dataStats && dataStats.columns && (
            <div className="flex items-center space-x-2 text-sm">
              <TrendingUp className="w-4 h-4 text-watson-600" />
              <span className="text-gray-600">
                {dataStats.columns} features
              </span>
            </div>
          )}
          
          {modelStats.totalModels > 0 && (
            <div className="flex items-center space-x-2 text-sm">
              <Brain className="w-4 h-4 text-watson-600" />
              <span className="text-gray-600">
                {modelStats.trainedModels} models
              </span>
            </div>
          )}
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-3">
          <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors relative">
            <Bell className="w-5 h-5 text-gray-600" />
            <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
          </button>
          
          <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors">
            <Settings className="w-5 h-5 text-gray-600" />
          </button>
          
          <div className="flex items-center space-x-2 pl-3 border-l border-gray-200">
            <div className="w-8 h-8 bg-gradient-to-r from-watson-600 to-watson-700 rounded-full flex items-center justify-center">
              <User className="w-4 h-4 text-white" />
            </div>
            <div className="hidden sm:block">
              <p className="text-sm font-medium text-gray-900">Admin User</p>
              <p className="text-xs text-gray-500">Administrator</p>
            </div>
          </div>
        </div>
      </div>
    </motion.nav>
  );
};

export default Navbar;
