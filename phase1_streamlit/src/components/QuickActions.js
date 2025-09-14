import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Brain, 
  BarChart3, 
  Download,
  Settings,
  Play
} from 'lucide-react';

const QuickActions = () => {
  const actions = [
    {
      title: "Upload Data",
      description: "Upload CSV dataset for analysis",
      icon: Upload,
      path: "/upload",
      color: "from-blue-500 to-blue-600",
      bgColor: "bg-blue-50",
      textColor: "text-blue-700"
    },
    {
      title: "Train Models",
      description: "Start AI model training",
      icon: Brain,
      path: "/training",
      color: "from-purple-500 to-purple-600",
      bgColor: "bg-purple-50",
      textColor: "text-purple-700"
    },
    {
      title: "View Results",
      description: "Check model performance",
      icon: BarChart3,
      path: "/results",
      color: "from-green-500 to-green-600",
      bgColor: "bg-green-50",
      textColor: "text-green-700"
    },
    {
      title: "Export Report",
      description: "Download analysis report",
      icon: Download,
      path: "/results",
      color: "from-orange-500 to-orange-600",
      bgColor: "bg-orange-50",
      textColor: "text-orange-700"
    }
  ];

  return (
    <div className="card">
      <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
        <Play className="w-5 h-5 mr-2 text-watson-600" />
        Quick Actions
      </h3>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {actions.map((action, index) => {
          const Icon = action.icon;
          return (
            <motion.div
              key={action.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Link
                to={action.path}
                className={`block p-4 rounded-lg border border-gray-200 hover:shadow-md transition-all duration-200 ${action.bgColor}`}
              >
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg bg-gradient-to-r ${action.color}`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <h4 className={`font-semibold ${action.textColor}`}>
                      {action.title}
                    </h4>
                    <p className="text-sm text-gray-600 mt-1">
                      {action.description}
                    </p>
                  </div>
                </div>
              </Link>
            </motion.div>
          );
        })}
      </div>
      
      <div className="mt-6 pt-4 border-t border-gray-200">
        <Link
          to="/settings"
          className="inline-flex items-center text-sm text-gray-600 hover:text-watson-700 transition-colors"
        >
          <Settings className="w-4 h-4 mr-2" />
          Advanced Settings
        </Link>
      </div>
    </div>
  );
};

export default QuickActions;
