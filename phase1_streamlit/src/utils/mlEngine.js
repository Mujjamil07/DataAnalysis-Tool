// Real Machine Learning Engine
// This handles actual data processing, model training, and metrics calculation

export class MLEngine {
  constructor() {
    this.models = {};
    this.results = {};
  }

  // Real data preprocessing
  preprocessData(rawData, targetColumn) {
    console.log('Starting real data preprocessing...');
    
    // Convert data to proper format
    const features = [];
    const labels = [];
    const featureNames = [];
    
    // Extract feature names (excluding target)
    Object.keys(rawData[0]).forEach(col => {
      if (col !== targetColumn) {
        featureNames.push(col);
      }
    });

    // Process each row
    rawData.forEach(row => {
      const featureRow = [];
      
      featureNames.forEach(feature => {
        let value = row[feature];
        
        // Handle different data types
        if (typeof value === 'string') {
          // Convert categorical to numeric
          value = this.categoricalToNumeric(value);
        } else if (typeof value === 'number') {
          // Keep numeric as is
        } else {
          value = 0; // Default for null/undefined
        }
        
        featureRow.push(value);
      });
      
      // Handle target variable
      let label = row[targetColumn];
      if (typeof label === 'string') {
        // Convert string labels to binary
        const uniqueLabels = [...new Set(rawData.map(r => r[targetColumn]))];
        label = uniqueLabels.indexOf(label) > uniqueLabels.length / 2 ? 1 : 0;
      } else if (typeof label === 'number') {
        label = label > 0.5 ? 1 : 0;
      } else {
        label = 0;
      }
      
      features.push(featureRow);
      labels.push(label);
    });

    // Normalize features
    const normalizedFeatures = this.normalizeFeatures(features);
    
    return {
      features: normalizedFeatures,
      labels: labels,
      featureNames: featureNames,
      originalData: rawData
    };
  }

  // Convert categorical variables to numeric
  categoricalToNumeric(value) {
    // Simple hash-based conversion
    return value.split('').reduce((sum, char) => sum + char.charCodeAt(0), 0) / 1000;
  }

  // Normalize features to [0, 1] range
  normalizeFeatures(features) {
    if (features.length === 0) return features;
    
    const numFeatures = features[0].length;
    const normalized = [];
    
    for (let j = 0; j < numFeatures; j++) {
      const column = features.map(row => row[j]);
      const min = Math.min(...column);
      const max = Math.max(...column);
      const range = max - min;
      
      for (let i = 0; i < features.length; i++) {
        if (!normalized[i]) normalized[i] = [];
        normalized[i][j] = range === 0 ? 0 : (features[i][j] - min) / range;
      }
    }
    
    return normalized;
  }

  // Real train-test split
  trainTestSplit(features, labels, testSize = 0.2) {
    const totalSamples = features.length;
    const testSamples = Math.floor(totalSamples * testSize);
    
    // Shuffle indices
    const indices = Array.from({length: totalSamples}, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    // Split data
    const testIndices = indices.slice(0, testSamples);
    const trainIndices = indices.slice(testSamples);
    
    const X_train = trainIndices.map(i => features[i]);
    const X_test = testIndices.map(i => features[i]);
    const y_train = trainIndices.map(i => labels[i]);
    const y_test = testIndices.map(i => labels[i]);
    
    return { X_train, X_test, y_train, y_test };
  }

  // Real Logistic Regression implementation
  trainLogisticRegression(X_train, y_train) {
    console.log('Training Logistic Regression...');
    
    const learningRate = 0.01;
    const iterations = 200;
    const lambda = 0.01; // L2 regularization
    
    const numFeatures = X_train[0].length;
    const weights = new Array(numFeatures).fill(0);
    let bias = 0;
    
    // Gradient descent with regularization
    for (let iter = 0; iter < iterations; iter++) {
      for (let i = 0; i < X_train.length; i++) {
        const z = weights.reduce((sum, w, j) => sum + w * X_train[i][j], 0) + bias;
        const prediction = 1 / (1 + Math.exp(-z));
        const error = y_train[i] - prediction;
        
        // Update weights with L2 regularization
        for (let j = 0; j < weights.length; j++) {
          const gradient = error * X_train[i][j] + lambda * weights[j];
          weights[j] += learningRate * gradient;
        }
        bias += learningRate * error;
      }
    }
    
    return { weights, bias };
  }

  // Real Random Forest implementation
  trainRandomForest(X_train, y_train) {
    console.log('Training Random Forest...');
    
    const nTrees = Math.min(20, Math.max(10, Math.floor(X_train.length / 50)));
    const trees = [];
    
    for (let tree = 0; tree < nTrees; tree++) {
      // Bootstrap sampling
      const bootstrapIndices = [];
      for (let i = 0; i < X_train.length; i++) {
        bootstrapIndices.push(Math.floor(Math.random() * X_train.length));
      }
      
      // Feature bagging
      const featureIndices = [];
      for (let i = 0; i < X_train[0].length; i++) {
        if (Math.random() < 0.7) featureIndices.push(i);
      }
      
      // Simple decision tree
      const treeModel = this.buildDecisionTree(
        bootstrapIndices.map(i => X_train[i]),
        bootstrapIndices.map(i => y_train[i]),
        featureIndices
      );
      
      trees.push(treeModel);
    }
    
    return { trees, featureIndices: trees[0]?.featureIndices || [] };
  }

  // Build simple decision tree
  buildDecisionTree(X, y, featureIndices) {
    if (X.length === 0) return null;
    
    const maxDepth = 5;
    const minSamples = 2;
    
    const buildNode = (X, y, depth, featureIndices) => {
      if (depth >= maxDepth || X.length < minSamples) {
        return { type: 'leaf', prediction: y.reduce((a, b) => a + b, 0) / y.length > 0.5 ? 1 : 0 };
      }
      
      // Find best split
      let bestFeature = -1;
      let bestThreshold = 0;
      let bestGini = Infinity;
      
      featureIndices.forEach(featureIndex => {
        const values = X.map(row => row[featureIndex]);
        const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
        
        for (let i = 0; i < uniqueValues.length - 1; i++) {
          const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
          
          const leftIndices = X.map((row, idx) => row[featureIndex] <= threshold ? idx : -1).filter(idx => idx !== -1);
          const rightIndices = X.map((row, idx) => row[featureIndex] > threshold ? idx : -1).filter(idx => idx !== -1);
          
          if (leftIndices.length === 0 || rightIndices.length === 0) continue;
          
          const leftY = leftIndices.map(idx => y[idx]);
          const rightY = rightIndices.map(idx => y[idx]);
          
          const gini = this.calculateGini(leftY) * leftY.length / y.length + 
                      this.calculateGini(rightY) * rightY.length / y.length;
          
          if (gini < bestGini) {
            bestGini = gini;
            bestFeature = featureIndex;
            bestThreshold = threshold;
          }
        }
      });
      
      if (bestFeature === -1) {
        return { type: 'leaf', prediction: y.reduce((a, b) => a + b, 0) / y.length > 0.5 ? 1 : 0 };
      }
      
      const leftIndices = X.map((row, idx) => row[bestFeature] <= bestThreshold ? idx : -1).filter(idx => idx !== -1);
      const rightIndices = X.map((row, idx) => row[bestFeature] > bestThreshold ? idx : -1).filter(idx => idx !== -1);
      
      return {
        type: 'split',
        feature: bestFeature,
        threshold: bestThreshold,
        left: buildNode(leftIndices.map(idx => X[idx]), leftIndices.map(idx => y[idx]), depth + 1, featureIndices),
        right: buildNode(rightIndices.map(idx => X[idx]), rightIndices.map(idx => y[idx]), depth + 1, featureIndices)
      };
    };
    
    return {
      root: buildNode(X, y, 0, featureIndices),
      featureIndices: featureIndices
    };
  }

  // Calculate Gini impurity
  calculateGini(y) {
    if (y.length === 0) return 0;
    const p1 = y.reduce((sum, val) => sum + val, 0) / y.length;
    const p0 = 1 - p1;
    return 1 - p0 * p0 - p1 * p1;
  }

  // Make predictions with Logistic Regression
  predictLogisticRegression(model, X) {
    const { weights, bias } = model;
    return X.map(features => {
      const z = weights.reduce((sum, w, j) => sum + w * features[j], 0) + bias;
      const probability = 1 / (1 + Math.exp(-z));
      return probability > 0.5 ? 1 : 0;
    });
  }

  // Make predictions with Random Forest
  predictRandomForest(model, X) {
    const { trees } = model;
    
    return X.map(features => {
      const predictions = trees.map(tree => this.predictTree(tree, features));
      const avgPrediction = predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
      return avgPrediction > 0.5 ? 1 : 0;
    });
  }

  // Predict with single tree
  predictTree(tree, features) {
    const traverse = (node) => {
      if (node.type === 'leaf') {
        return node.prediction;
      }
      
      const featureValue = features[node.feature];
      if (featureValue <= node.threshold) {
        return traverse(node.left);
      } else {
        return traverse(node.right);
      }
    };
    
    return traverse(tree.root);
  }

  // Calculate real metrics
  calculateMetrics(yTrue, yPred) {
    console.log('Calculating real metrics...');
    
    let truePositives = 0;
    let trueNegatives = 0;
    let falsePositives = 0;
    let falseNegatives = 0;
    
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] === 1 && yPred[i] === 1) truePositives++;
      else if (yTrue[i] === 0 && yPred[i] === 0) trueNegatives++;
      else if (yTrue[i] === 0 && yPred[i] === 1) falsePositives++;
      else if (yTrue[i] === 1 && yPred[i] === 0) falseNegatives++;
    }
    
    // Real formulas
    const accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);
    const precision = truePositives / (truePositives + falsePositives) || 0;
    const recall = truePositives / (truePositives + falseNegatives) || 0;
    const f1Score = (2 * precision * recall) / (precision + recall) || 0;
    
    // Calculate ROC-AUC
    const auc = this.calculateAUC(yTrue, yPred);
    
    return {
      accuracy: Math.round(accuracy * 1000) / 1000,
      precision: Math.round(precision * 1000) / 1000,
      recall: Math.round(recall * 1000) / 1000,
      f1Score: Math.round(f1Score * 1000) / 1000,
      auc: Math.round(auc * 1000) / 1000,
      confusionMatrix: {
        truePositives,
        trueNegatives,
        falsePositives,
        falseNegatives
      }
    };
  }

  // Calculate ROC-AUC
  calculateAUC(yTrue, yPred) {
    // Simplified AUC calculation
    const totalPositive = yTrue.reduce((sum, val) => sum + val, 0);
    const totalNegative = yTrue.length - totalPositive;
    
    if (totalPositive === 0 || totalNegative === 0) return 0.5;
    
    let auc = 0;
    for (let i = 0; i < yTrue.length; i++) {
      for (let j = 0; j < yTrue.length; j++) {
        if (yTrue[i] === 1 && yTrue[j] === 0) {
          if (yPred[i] > yPred[j]) auc += 1;
          else if (yPred[i] === yPred[j]) auc += 0.5;
        }
      }
    }
    
    return auc / (totalPositive * totalNegative);
  }

  // Main training function
  async trainModel(modelType, data, targetColumn) {
    console.log(`Training ${modelType} with real data...`);
    
    // Preprocess data
    const processedData = this.preprocessData(data, targetColumn);
    
    // Train-test split
    const { X_train, X_test, y_train, y_test } = this.trainTestSplit(
      processedData.features, 
      processedData.labels
    );
    
    let model;
    let predictions;
    
    // Train model based on type
    switch (modelType) {
      case 'logistic_regression':
        model = this.trainLogisticRegression(X_train, y_train);
        predictions = this.predictLogisticRegression(model, X_test);
        break;
      case 'random_forest':
        model = this.trainRandomForest(X_train, y_train);
        predictions = this.predictRandomForest(model, X_test);
        break;
      default:
        throw new Error(`Unknown model type: ${modelType}`);
    }
    
    // Calculate metrics
    const metrics = this.calculateMetrics(y_test, predictions);
    
    return {
      model,
      metrics,
      testData: { X_test, y_test },
      predictions,
      featureNames: processedData.featureNames
    };
  }
}

export default MLEngine;
