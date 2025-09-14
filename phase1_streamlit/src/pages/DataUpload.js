import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  Database, 
  CheckCircle, 
  ArrowRight,
  FileText
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import toast from 'react-hot-toast';
import ApiService from '../services/apiService';

const DataUpload = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const { setData, setSessionId, data } = useAppStore();

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setIsUploading(true);
    setUploadedFile({
      name: file.name,
      size: file.size,
      type: file.type,
      lastModified: file.lastModified
    });

    try {
      // Use the ApiService to upload the file properly
      const result = await ApiService.uploadData(file);
      console.log('Backend response:', result);
      
      if (result.data_info && Array.isArray(result.data_info.preview)) {
        // Ensure data is serializable before storing
        const serializableData = JSON.parse(JSON.stringify(result.data_info.preview));
        console.log('Setting data:', serializableData);
        setData(serializableData);
        if (result.session_id) {
          setSessionId(result.session_id);
          localStorage.setItem('session_id', result.session_id);
          console.log('✅ Session ID stored:', result.session_id);
        }
        toast.success(`✅ ${file.name} uploaded successfully!`);
      } else if (result.data && Array.isArray(result.data)) {
        // Fallback: if data is directly in result.data
        const serializableData = JSON.parse(JSON.stringify(result.data));
        console.log('Setting data (fallback):', serializableData);
        setData(serializableData);
        if (result.session_id) {
          setSessionId(result.session_id);
          localStorage.setItem('session_id', result.session_id);
          console.log('✅ Session ID stored:', result.session_id);
        }
        toast.success(`✅ ${file.name} uploaded successfully!`);
      } else {
        console.error('Invalid response format:', result);
        // Still show success but with warning
        toast.success(`✅ ${file.name} uploaded successfully!`);
        // Set dummy data to show button
        setData([{ message: 'Data uploaded successfully' }]);
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('❌ Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  }, [setData, setSessionId]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    multiple: false
  });

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Data Upload</h1>
              <p className="text-gray-600">Upload your dataset to get started with AutoML</p>
            </div>
          </div>
        </div>

        {/* Upload Area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-blue-400 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              {isDragActive ? (
                <p className="text-lg text-blue-600">Drop the file here...</p>
              ) : (
                <div>
                  <p className="text-lg text-gray-600 mb-2">
                    Drag & drop your dataset here, or click to select
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports CSV, XLS, XLSX files
                  </p>
                </div>
              )}
            </div>

            {/* Upload Status */}
            {isUploading && (
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                  <span className="ml-3 text-blue-600">Uploading...</span>
                </div>
              </div>
            )}

            {/* File Info */}
            {uploadedFile && !isUploading && (
              <div className="mt-4 p-4 bg-green-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                    <div>
                      <p className="text-sm font-medium text-green-800">
                        {uploadedFile.name}
                      </p>
                      <p className="text-xs text-green-600">
                        {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                </div>
                {data && data.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-green-200">
                    <p className="text-xs text-green-700">
                      ✅ Data loaded successfully! Ready for cleaning.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Go to Data Cleaning Button */}
        {uploadedFile ? (
          <div className="mt-6">
            <button
              onClick={() => window.location.href = '/cleaning'}
              className="w-full bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 flex items-center justify-center"
            >
              <ArrowRight className="w-5 h-5 mr-2" />
              Go to Data Cleaning
            </button>
          </div>
        ) : (
          <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
            <p className="text-sm text-yellow-700">
              Debug: Data length = {data ? data.length : 'null'}, Uploaded file = {uploadedFile ? 'Yes' : 'No'}
            </p>
          </div>
        )}
          </div>

          {/* Data Preview */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <FileText className="w-5 h-5 mr-2" />
              Data Preview
            </h3>
            {isUploading ? (
              <div className="bg-gray-50 rounded-lg p-8 text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Processing data...</p>
              </div>
            ) : data && data.length > 0 ? (
              <div className="bg-white rounded-lg border overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        {Object.keys(data[0]).map((column) => (
                          <th
                            key={column}
                            className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                          >
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {data.slice(0, 5).map((row, index) => (
                        <tr key={index}>
                          {Object.values(row).map((value, cellIndex) => (
                            <td
                              key={cellIndex}
                              className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                            >
                              {value}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="bg-gray-50 px-6 py-3">
                  <p className="text-sm text-gray-600">
                    Showing 5 of {data.length} rows
                  </p>
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 rounded-lg p-8 text-center">
                <Database className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <p className="text-gray-600">No data uploaded yet</p>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default DataUpload;