import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  Upload, 
  Brain, 
  BarChart3, 
  Database, 
  TrendingUp, 
  Users, 
  Activity,
  ArrowRight,
  Play,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import MetricCard from '../components/MetricCard';
import QuickActions from '../components/QuickActions';
import RecentActivity from '../components/RecentActivity';

const Dashboard = () => {
  const { data, models, results, getDataStats, getModelStats } = useAppStore();
  const dataStats = getDataStats();
  const modelStats = getModelStats();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="p-6 space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="text-center">
        <h1 className="text-4xl font-bold text-gradient font-ibm mb-2">
          Welcome to Watson AI Studio
        </h1>
        <p className="text-lg text-gray-600 font-inter">
          Professional Data Analysis & Machine Learning Platform
        </p>
      </motion.div>

      {/* Key Metrics */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Dataset Size"
          value={dataStats ? dataStats.rows.toLocaleString() : "0"}
          subtitle={dataStats ? "Total records" : "No data uploaded"}
          icon={Database}
          status={data ? "success" : "warning"}
        />
        
        <MetricCard
          title="Features"
          value={dataStats ? dataStats.columns.toString() : "0"}
          subtitle={dataStats ? "Total columns" : "No data uploaded"}
          icon={TrendingUp}
          status={data ? "info" : "warning"}
        />
        
        <MetricCard
          title="Models Trained"
          value={modelStats.trainedModels.toString()}
          subtitle="Ready for predictions"
          icon={Brain}
          status={modelStats.trainedModels > 0 ? "success" : "warning"}
        />
        
        <MetricCard
          title="Best Model"
          value={modelStats.bestModel || "N/A"}
          subtitle="Top performer"
          icon={CheckCircle}
          status={modelStats.bestModel ? "primary" : "warning"}
        />
      </motion.div>

      {/* Quick Actions */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <QuickActions />
        
        {/* Status Overview */}
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-watson-600" />
            System Status
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <div className="flex items-center">
                <CheckCircle className="w-5 h-5 text-green-600 mr-3" />
                <span className="font-medium text-green-800">AI Engine</span>
              </div>
              <span className="text-sm text-green-600">Online</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center">
                <Database className="w-5 h-5 text-blue-600 mr-3" />
                <span className="font-medium text-blue-800">Data Processing</span>
              </div>
              <span className="text-sm text-blue-600">Ready</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
              <div className="flex items-center">
                <Brain className="w-5 h-5 text-purple-600 mr-3" />
                <span className="font-medium text-purple-800">Model Training</span>
              </div>
              <span className="text-sm text-purple-600">Available</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-orange-50 rounded-lg">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-orange-600 mr-3" />
                <span className="font-medium text-orange-800">Storage</span>
              </div>
              <span className="text-sm text-orange-600">85% Used</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Recent Activity & Getting Started */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentActivity />
        
        {/* Getting Started */}
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Play className="w-5 h-5 mr-2 text-watson-600" />
            Getting Started
          </h3>
          
          <div className="space-y-4">
            <div className="p-4 border border-gray-200 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">1. Upload Your Data</h4>
              <p className="text-sm text-gray-600 mb-3">
                Start by uploading your CSV dataset. Watson AI will automatically analyze and clean your data.
              </p>
              <Link to="/upload" className="btn-primary inline-flex items-center text-sm">
                Upload Data
                <ArrowRight className="w-4 h-4 ml-2" />
              </Link>
            </div>
            
            <div className="p-4 border border-gray-200 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">2. Train AI Models</h4>
              <p className="text-sm text-gray-600 mb-3">
                Let Watson AI automatically train multiple machine learning models on your data.
              </p>
              <Link to="/training" className="btn-secondary inline-flex items-center text-sm">
                Start Training
                <ArrowRight className="w-4 h-4 ml-2" />
              </Link>
            </div>
            
            <div className="p-4 border border-gray-200 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">3. View Results</h4>
              <p className="text-sm text-gray-600 mb-3">
                Compare model performance and download detailed analysis reports.
              </p>
              <Link to="/results" className="btn-secondary inline-flex items-center text-sm">
                View Results
                <ArrowRight className="w-4 h-4 ml-2" />
              </Link>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Dashboard;
