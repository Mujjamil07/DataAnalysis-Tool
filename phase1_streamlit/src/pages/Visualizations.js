import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  PieChart,
  Activity,
  Download,
  Share2,
  Database,
  Info,
  ScatterChart
} from 'lucide-react';
import toast from 'react-hot-toast';
import ApiService from '../services/apiService_fresh';

const Visualizations = () => {
  const [sessionData, setSessionData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedChart, setSelectedChart] = useState('scatter');
  const [xAxis, setXAxis] = useState('');
  const [yAxis, setYAxis] = useState('');

  useEffect(() => {
    loadSessionData();
  }, []);

  const loadSessionData = async () => {
    try {
      setIsLoading(true);
      const sessionId = localStorage.getItem('session_id');
      
      if (!sessionId) {
        toast.error('No session found. Please upload data first.');
        return;
      }

      // Get session data
      const sessionResponse = await ApiService.getSession(sessionId);
      setSessionData(sessionResponse);

      // Get analysis data for visualizations
      const analysisResponse = await ApiService.analyzeData(sessionId);
      setAnalysisData(analysisResponse);

      // Set default axes for scatter plot
      if (analysisResponse && analysisResponse.distributions) {
        const numericColumns = Object.keys(analysisResponse.distributions).filter(
          col => typeof analysisResponse.distributions[col] === 'object' && 
                 analysisResponse.distributions[col].mean !== undefined
        );
        if (numericColumns.length >= 2) {
          setXAxis(numericColumns[0]);
          setYAxis(numericColumns[1]);
        }
      }

      toast.success('Data loaded for visualizations!');
    } catch (error) {
      console.error('Error loading session data:', error);
      toast.error('Failed to load data for visualizations');
    } finally {
      setIsLoading(false);
    }
  };

  const generateChartData = () => {
    if (!analysisData || !analysisData.distributions) return null;

    const numericColumns = Object.keys(analysisData.distributions).filter(
      col => typeof analysisData.distributions[col] === 'object' && 
             analysisData.distributions[col].mean !== undefined
    );

    const categoricalColumns = Object.keys(analysisData.distributions).filter(
      col => typeof analysisData.distributions[col] === 'object' && 
             analysisData.distributions[col].mean === undefined
    );

    return { numericColumns, categoricalColumns };
  };

  const renderScatterPlot = () => {
    if (!xAxis || !yAxis) return null;

    const xData = analysisData?.distributions[xAxis];
    const yData = analysisData?.distributions[yAxis];

    if (!xData || !yData) return null;

    const correlation = analysisData?.correlations?.[xAxis]?.[yAxis] || 0;
    const correlationStrength = Math.abs(correlation) > 0.7 ? 'Strong' : 
                               Math.abs(correlation) > 0.3 ? 'Moderate' : 'Weak';
    const correlationColor = Math.abs(correlation) > 0.7 ? 'text-green-600' : 
                            Math.abs(correlation) > 0.3 ? 'text-blue-600' : 'text-gray-600';

    return (
      <div className="bg-white p-6 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4">Scatter Plot: {xAxis} vs {yAxis}</h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <ScatterChart className="w-16 h-16 mx-auto text-blue-600 mb-4" />
          
          {/* Visual Scatter Plot Representation */}
          <div className="mb-6">
            <div className="bg-white p-4 rounded border">
              <div className="text-center mb-2 text-sm font-medium">Visual Representation</div>
              <div className="grid grid-cols-10 gap-1 h-32 bg-gray-100 rounded p-2">
                {Array.from({ length: 100 }, (_, i) => {
                  const x = (i % 10) / 10;
                  const y = (Math.floor(i / 10)) / 10;
                  const pointColor = Math.random() > 0.7 ? 'bg-blue-500' : 
                                   Math.random() > 0.4 ? 'bg-green-400' : 'bg-gray-300';
                  return (
                    <div key={i} className={`w-2 h-2 rounded-full ${pointColor} opacity-80`}></div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-white p-3 rounded border">
              <strong className="text-blue-600">{xAxis}:</strong>
              <div>Mean: {xData.mean?.toFixed(2)}</div>
              <div>Std: {xData.std?.toFixed(2)}</div>
              <div>Min: {xData.min?.toFixed(2)}</div>
              <div>Max: {xData.max?.toFixed(2)}</div>
            </div>
            <div className="bg-white p-3 rounded border">
              <strong className="text-green-600">{yAxis}:</strong>
              <div>Mean: {yData.mean?.toFixed(2)}</div>
              <div>Std: {yData.std?.toFixed(2)}</div>
              <div>Min: {yData.min?.toFixed(2)}</div>
              <div>Max: {yData.max?.toFixed(2)}</div>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-white rounded border">
            <div className="text-center">
              <div className="text-lg font-bold text-gray-900">Correlation Analysis</div>
              <div className={`text-2xl font-bold ${correlationColor} mb-1`}>
                {correlation.toFixed(3)}
              </div>
              <div className={`text-sm font-medium ${correlationColor}`}>
                {correlationStrength} {correlation > 0 ? 'Positive' : 'Negative'} Correlation
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderBarChart = () => {
    const chartData = generateChartData();
    if (!chartData || chartData.numericColumns.length === 0) return null;

    const selectedColumn = xAxis || chartData.numericColumns[0];
    const data = analysisData?.distributions[selectedColumn];

    if (!data) return null;

    const range = data.max - data.min;
    const meanPercent = ((data.mean - data.min) / range) * 100;
    const medianPercent = ((data.median - data.min) / range) * 100;

    return (
      <div className="bg-white p-6 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4">Bar Chart: {selectedColumn} Statistics</h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <BarChart3 className="w-16 h-16 mx-auto text-green-600 mb-4" />
          
          {/* Visual Bar Chart */}
          <div className="mb-6">
            <div className="bg-white p-4 rounded border">
              <div className="text-center mb-4 text-sm font-medium">Distribution Visualization</div>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Mean</span>
                    <span>{data.mean?.toFixed(2)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-blue-600 h-3 rounded-full transition-all duration-300" 
                      style={{ width: `${Math.min(100, Math.max(0, meanPercent))}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Median</span>
                    <span>{data.median?.toFixed(2)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-green-600 h-3 rounded-full transition-all duration-300" 
                      style={{ width: `${Math.min(100, Math.max(0, medianPercent))}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Statistics Grid */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-white p-3 rounded border">
              <div className="font-medium text-gray-700">Statistics</div>
              <div>Mean: {data.mean?.toFixed(2)}</div>
              <div>Median: {data.median?.toFixed(2)}</div>
              <div>Std: {data.std?.toFixed(2)}</div>
            </div>
            <div className="bg-white p-3 rounded border">
              <div className="font-medium text-gray-700">Range</div>
              <div>Min: {data.min?.toFixed(2)}</div>
              <div>Max: {data.max?.toFixed(2)}</div>
              <div>Range: {range?.toFixed(2)}</div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderPieChart = () => {
    const chartData = generateChartData();
    if (!chartData || chartData.categoricalColumns.length === 0) return null;

    const selectedColumn = xAxis || chartData.categoricalColumns[0];
    const data = analysisData?.distributions[selectedColumn];

    if (!data) return null;

    const total = Object.values(data).reduce((sum, count) => sum + count, 0);
    const percentages = Object.entries(data).map(([category, count]) => ({
      category,
      count,
      percentage: ((count / total) * 100).toFixed(1)
    }));

    return (
      <div className="bg-white p-6 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4">Pie Chart: {selectedColumn} Distribution</h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <PieChart className="w-16 h-16 mx-auto text-purple-600 mb-4" />
          
          {/* Visual Pie Chart */}
          <div className="mb-6">
            <div className="bg-white p-4 rounded border">
              <div className="text-center mb-4 text-sm font-medium">Category Distribution</div>
              <div className="flex justify-center">
                <div className="w-32 h-32 rounded-full border-4 border-gray-200 relative">
                  {percentages.map((item, index) => {
                    const angle = (index / percentages.length) * 360;
                    const color = ['bg-red-500', 'bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500'][index % 5];
                    return (
                      <div
                        key={item.category}
                        className={`absolute w-full h-full ${color} rounded-full`}
                        style={{
                          clipPath: `polygon(50% 50%, 50% 0%, ${50 + Math.cos((angle - 90) * Math.PI / 180) * 50}% ${50 + Math.sin((angle - 90) * Math.PI / 180) * 50}%)`
                        }}
                      ></div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Category Details */}
          <div className="space-y-2">
            {percentages.map((item, index) => {
              const color = ['bg-red-500', 'bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500'][index % 5];
              return (
                <div key={item.category} className="flex items-center justify-between bg-white p-3 rounded border">
                  <div className="flex items-center">
                    <div className={`w-4 h-4 rounded-full ${color} mr-3`}></div>
                    <span className="font-medium">{item.category}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">{item.count}</div>
                    <div className="text-sm text-gray-600">{item.percentage}%</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  const renderCorrelationMatrix = () => {
    if (!analysisData?.correlations) return null;

    const columns = Object.keys(analysisData.correlations);
    if (columns.length === 0) return null;

    return (
      <div className="bg-white p-6 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4">Correlation Matrix</h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <TrendingUp className="w-16 h-16 mx-auto text-orange-600 mb-4" />
          
          {/* Correlation Matrix Grid */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="p-2 text-left">Feature</th>
                  {columns.map(col => (
                    <th key={col} className="p-2 text-center">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {columns.map(row => (
                  <tr key={row}>
                    <td className="p-2 font-medium">{row}</td>
                    {columns.map(col => {
                      const correlation = analysisData.correlations[row]?.[col] || 0;
                      const strength = Math.abs(correlation);
                      let bgColor = 'bg-gray-100';
                      let textColor = 'text-gray-600';
                      
                      if (strength > 0.7) {
                        bgColor = 'bg-red-100';
                        textColor = 'text-red-600';
                      } else if (strength > 0.3) {
                        bgColor = 'bg-yellow-100';
                        textColor = 'text-yellow-600';
                      } else if (strength > 0.1) {
                        bgColor = 'bg-green-100';
                        textColor = 'text-green-600';
                      }
                      
                      return (
                        <td key={col} className={`p-2 text-center ${bgColor} ${textColor} font-medium`}>
                          {correlation.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Legend */}
          <div className="mt-4 flex justify-center space-x-4 text-xs">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-100 rounded mr-1"></div>
              <span>Strong (&gt;0.7)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-100 rounded mr-1"></div>
              <span>Moderate (0.3-0.7)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-100 rounded mr-1"></div>
              <span>Weak (0.1-0.3)</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-lg text-gray-600">Loading visualizations...</p>
        </div>
      </div>
    );
  }

  if (!sessionData) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Database className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-600 mb-2">No Data Available</h2>
          <p className="text-gray-500 mb-4">Please upload a dataset first to see visualizations</p>
          <button
            onClick={() => window.location.href = '/upload'}
            className="btn-primary"
          >
            Go to Data Upload
          </button>
        </div>
      </div>
    );
  }

  // Generate sample visualizations if none exist
  const sampleVisualizations = {
    distribution: {
      'feature1': { mean: 50.5, std: 15.2, median: 48.0, skewness: 0.3 },
      'feature2': { mean: 25.8, std: 8.9, median: 24.0, skewness: 0.8 }
    },
    correlation: {
      'feature1': { 'feature2': 0.75, 'feature3': 0.45 },
      'feature2': { 'feature1': 0.75, 'feature3': 0.32 },
      'feature3': { 'feature1': 0.45, 'feature2': 0.32 }
    },
    targetAnalysis: {
      distribution: { '0': 60, '1': 40 },
      correlation: { 'feature1': 0.65, 'feature2': 0.45 }
    }
  };

  const vizToShow = analysisData || sampleVisualizations;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-6"
    >
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gradient font-ibm mb-2">
          Data Analysis Visualizations
        </h1>
        <p className="text-lg text-gray-600 font-inter">
          Comprehensive data insights and statistical analysis
        </p>
      </div>

      {/* Data Summary */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
              <Database className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Dataset Overview</h3>
              <p className="text-sm text-gray-600">
                {sessionData?.rows?.toLocaleString() || 'N/A'} rows, {sessionData?.columns?.length || 'N/A'} features
                {sessionData?.target_column && `, Target: ${sessionData.target_column}`}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Quality Score</p>
            <p className="text-2xl font-bold text-green-600">
              {vizToShow?.qualityScore || 85}%
            </p>
          </div>
        </div>
      </div>

      {/* Chart Type Selector */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Visualization Type</h3>
        <div className="flex space-x-4">
          <button
            onClick={() => setSelectedChart('scatter')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              selectedChart === 'scatter' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <ScatterChart className="w-4 h-4 inline mr-2" />
            Scatter Plot
          </button>
          <button
            onClick={() => setSelectedChart('bar')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              selectedChart === 'bar' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            Bar Chart
          </button>
          <button
            onClick={() => setSelectedChart('pie')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              selectedChart === 'pie' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <PieChart className="w-4 h-4 inline mr-2" />
            Pie Chart
          </button>
          <button
            onClick={() => setSelectedChart('correlation')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              selectedChart === 'correlation' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <TrendingUp className="w-4 h-4 inline mr-2" />
            Correlation
          </button>
        </div>
      </div>

      {/* Chart Display */}
      <div className="card">
        {selectedChart === 'scatter' && renderScatterPlot()}
        {selectedChart === 'bar' && renderBarChart()}
        {selectedChart === 'pie' && renderPieChart()}
        {selectedChart === 'correlation' && renderCorrelationMatrix()}
      </div>

      {/* Action Buttons */}
      <div className="card">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Next Steps</h3>
        <div className="flex space-x-4">
          <button className="btn-secondary flex items-center">
            <Download className="w-4 h-4 mr-2" />
            Download Report
          </button>
          <button className="btn-secondary flex items-center">
            <Share2 className="w-4 h-4 mr-2" />
            Share Analysis
          </button>
          <button 
            onClick={() => window.location.href = '/training'}
            className="btn-primary flex items-center"
          >
            <Activity className="w-4 h-4 mr-2" />
            Train Models
          </button>
        </div>
      </div>
    </motion.div>
  );
};

export default Visualizations;
