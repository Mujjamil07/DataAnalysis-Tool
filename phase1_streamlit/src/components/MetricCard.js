import React from 'react';
import { motion } from 'framer-motion';

const MetricCard = ({ title, value, subtitle, icon: Icon, status = "primary" }) => {
  const statusColors = {
    success: "border-green-500 bg-green-50",
    warning: "border-yellow-500 bg-yellow-50", 
    info: "border-blue-500 bg-blue-50",
    primary: "border-watson-600 bg-watson-50",
    error: "border-red-500 bg-red-50"
  };

  const iconColors = {
    success: "text-green-600",
    warning: "text-yellow-600",
    info: "text-blue-600", 
    primary: "text-watson-600",
    error: "text-red-600"
  };

  return (
    <motion.div
      whileHover={{ y: -5 }}
      className={`metric-card ${statusColors[status]} animate-fade-in`}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h3 className="text-sm font-medium text-gray-600 mb-1">{title}</h3>
          <p className="text-2xl font-bold text-gray-900 mb-1">{value}</p>
          <p className="text-xs text-gray-500">{subtitle}</p>
        </div>
        <div className={`p-3 rounded-lg ${iconColors[status]}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </motion.div>
  );
};

export default MetricCard;
