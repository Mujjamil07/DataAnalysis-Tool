import React from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Brain, 
  CheckCircle, 
  AlertCircle,
  Clock,
  FileText
} from 'lucide-react';

const RecentActivity = () => {
  const activities = [
    {
      id: 1,
      type: 'upload',
      title: 'Dataset uploaded',
      description: 'customer_data.csv (1,234 records)',
      time: '2 minutes ago',
      status: 'success',
      icon: Upload
    },
    {
      id: 2,
      type: 'training',
      title: 'Model training completed',
      description: 'Random Forest - 94.2% accuracy',
      time: '15 minutes ago',
      status: 'success',
      icon: Brain
    },
    {
      id: 3,
      type: 'analysis',
      title: 'Data analysis finished',
      description: 'Missing values handled, outliers removed',
      time: '1 hour ago',
      status: 'success',
      icon: CheckCircle
    },
    {
      id: 4,
      type: 'warning',
      title: 'High memory usage detected',
      description: 'Consider reducing dataset size',
      time: '2 hours ago',
      status: 'warning',
      icon: AlertCircle
    },
    {
      id: 5,
      type: 'report',
      title: 'Report generated',
      description: 'Model performance analysis.pdf',
      time: '3 hours ago',
      status: 'success',
      icon: FileText
    }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return 'text-green-600 bg-green-100';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getIconColor = (status) => {
    switch (status) {
      case 'success':
        return 'text-green-600';
      case 'warning':
        return 'text-yellow-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="card">
      <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
        <Clock className="w-5 h-5 mr-2 text-watson-600" />
        Recent Activity
      </h3>
      
      <div className="space-y-4">
        {activities.map((activity, index) => {
          const Icon = activity.icon;
          return (
            <motion.div
              key={activity.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-start space-x-3 p-3 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <div className={`p-2 rounded-lg ${getStatusColor(activity.status)}`}>
                <Icon className={`w-4 h-4 ${getIconColor(activity.status)}`} />
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-medium text-gray-900">
                    {activity.title}
                  </h4>
                  <span className="text-xs text-gray-500">
                    {activity.time}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-1">
                  {activity.description}
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-200">
        <button className="text-sm text-watson-600 hover:text-watson-700 font-medium transition-colors">
          View all activity →
        </button>
      </div>
    </div>
  );
};

export default RecentActivity;
