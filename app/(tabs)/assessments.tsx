import React, { useState, useRef } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Dimensions, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Video, Camera, Play, Square, RotateCcw, Timer, Activity } from 'lucide-react-native';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';

const { width, height } = Dimensions.get('window');

interface AssessmentType {
  id: string;
  name: string;
  description: string;
  duration: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  icon: string;
  color: string;
}

interface AIAnalysisResult {
  exerciseType: string;
  score: number;
  reps: number;
  form: string;
  suggestions: string[];
  benchmarkComparison: string;
}

export default function AssessmentsScreen() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [selectedAssessment, setSelectedAssessment] = useState<AssessmentType | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AIAnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  const assessmentTypes: AssessmentType[] = [
    {
      id: '1',
      name: 'Vertical Jump',
      description: 'Measure explosive leg power',
      duration: '30 sec',
      difficulty: 'Medium',
      icon: '🦘',
      color: '#10B981'
    },
    {
      id: '2',
      name: 'Sprint Test',
      description: '40m acceleration test',
      duration: '15 sec',
      difficulty: 'Hard',
      icon: '🏃‍♂️',
      color: '#F97316'
    },
    {
      id: '3',
      name: 'Push-ups',
      description: 'Upper body strength assessment',
      duration: '60 sec',
      difficulty: 'Easy',
      icon: '💪',
      color: '#8B5CF6'
    },
    {
      id: '4',
      name: 'Shuttle Run',
      description: 'Agility and speed test',
      duration: '45 sec',
      difficulty: 'Hard',
      icon: '⚡',
      color: '#EF4444'
    },
    {
      id: '5',
      name: 'Sit-ups',
      description: 'Core strength evaluation',
      duration: '60 sec',
      difficulty: 'Medium',
      icon: '🎯',
      color: '#06B6D4'
    },
    {
      id: '6',
      name: 'Endurance Run',
      description: 'Cardiovascular fitness test',
      duration: '10 min',
      difficulty: 'Hard',
      icon: '🏃‍♀️',
      color: '#DC2626'
    }
  ];

  const startAssessment = async (assessment: AssessmentType) => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        Alert.alert('Camera permission required', 'Please allow camera access to record assessments.');
        return;
      }
    }
    setSelectedAssessment(assessment);
    setShowCamera(true);
  };

  const startRecording = () => {
    setIsRecording(true);
    // Simulate recording duration based on assessment
    const duration = selectedAssessment?.duration === '30 sec' ? 30000 : 
                    selectedAssessment?.duration === '15 sec' ? 15000 : 60000;
    
    setTimeout(() => {
      stopRecording();
    }, duration);
  };

  const stopRecording = () => {
    setIsRecording(false);
    setShowCamera(false);
    simulateAIAnalysis();
  };

  const simulateAIAnalysis = () => {
    setIsAnalyzing(true);
    
    // Simulate AI processing time
    setTimeout(() => {
      const mockResults: AIAnalysisResult = {
        exerciseType: selectedAssessment?.name || 'Assessment',
        score: Math.floor(Math.random() * 30) + 70, // Score between 70-100
        reps: selectedAssessment?.name === 'Push-ups' ? Math.floor(Math.random() * 20) + 25 :
              selectedAssessment?.name === 'Sit-ups' ? Math.floor(Math.random() * 25) + 30 :
              selectedAssessment?.name === 'Vertical Jump' ? Math.floor(Math.random() * 20) + 40 : 1,
        form: ['Excellent', 'Good', 'Fair'][Math.floor(Math.random() * 3)],
        suggestions: [
          'Maintain consistent rhythm',
          'Focus on full range of motion',
          'Keep core engaged throughout'
        ],
        benchmarkComparison: 'Above average for your age group'
      };
      
      setAnalysisResult(mockResults);
      setIsAnalyzing(false);
      setSelectedAssessment(null);
    }, 3000);
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Easy': return '#10B981';
      case 'Medium': return '#F59E0B';
      case 'Hard': return '#EF4444';
      default: return '#6B7280';
    }
  };

  if (showCamera) {
    return (
      <View style={styles.cameraContainer}>
        <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
          <View style={styles.cameraOverlay}>
            <View style={styles.cameraHeader}>
              <Text style={styles.assessmentTitle}>{selectedAssessment?.name}</Text>
              <TouchableOpacity 
                style={styles.flipButton}
                onPress={() => setFacing(facing === 'back' ? 'front' : 'back')}
              >
                <RotateCcw size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>
            
            <View style={styles.cameraCenter}>
              <View style={styles.recordingIndicator}>
                <Text style={styles.instructionText}>
                  {isRecording ? `Recording ${selectedAssessment?.name}...` : `Ready to record ${selectedAssessment?.name}`}
                </Text>
                <Text style={styles.durationText}>{selectedAssessment?.duration}</Text>
              </View>
            </View>

            <View style={styles.cameraControls}>
              <TouchableOpacity 
                style={styles.cancelButton}
                onPress={() => setShowCamera(false)}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.recordButton, isRecording && styles.recordingButton]}
                onPress={isRecording ? stopRecording : startRecording}
                disabled={isRecording}
              >
                {isRecording ? (
                  <Square size={32} color="#FFFFFF" />
                ) : (
                  <Play size={32} color="#FFFFFF" />
                )}
              </TouchableOpacity>

              <View style={styles.placeholder} />
            </View>
          </View>
        </CameraView>
      </View>
    );
  }

  if (isAnalyzing) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.analyzingContainer}>
          <Animated.View entering={FadeInUp.duration(800)} style={styles.analyzingContent}>
            <Activity size={48} color="#1E40AF" />
            <Text style={styles.analyzingTitle}>AI Analysis in Progress</Text>
            <Text style={styles.analyzingSubtitle}>
              Our AI is analyzing your {selectedAssessment?.name.toLowerCase()} performance...
            </Text>
            <View style={styles.analyzingSteps}>
              <Text style={styles.analyzingStep}>✓ Video processed</Text>
              <Text style={styles.analyzingStep}>✓ Movement detected</Text>
              <Text style={styles.analyzingStep}>⏳ Calculating metrics...</Text>
            </View>
          </Animated.View>
        </View>
      </SafeAreaView>
    );
  }

  if (analysisResult) {
    return (
      <SafeAreaView style={styles.container}>
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <Animated.View entering={FadeInDown.duration(800)} style={styles.resultsContainer}>
            <View style={styles.resultsHeader}>
              <Text style={styles.resultsTitle}>Assessment Complete!</Text>
              <TouchableOpacity 
                style={styles.doneButton}
                onPress={() => setAnalysisResult(null)}
              >
                <Text style={styles.doneButtonText}>Done</Text>
              </TouchableOpacity>
            </View>

            <LinearGradient
              colors={['#1E40AF', '#3B82F6']}
              style={styles.scoreCard}
            >
              <Text style={styles.scoreLabel}>Your Score</Text>
              <Text style={styles.scoreValue}>{analysisResult.score}/100</Text>
              <Text style={styles.benchmarkText}>{analysisResult.benchmarkComparison}</Text>
            </LinearGradient>

            <View style={styles.metricsContainer}>
              <View style={styles.metricCard}>
                <Text style={styles.metricLabel}>Repetitions</Text>
                <Text style={styles.metricValue}>{analysisResult.reps}</Text>
              </View>
              <View style={styles.metricCard}>
                <Text style={styles.metricLabel}>Form Quality</Text>
                <Text style={styles.metricValue}>{analysisResult.form}</Text>
              </View>
            </View>

            <View style={styles.suggestionsContainer}>
              <Text style={styles.suggestionsTitle}>AI Recommendations</Text>
              {analysisResult.suggestions.map((suggestion, index) => (
                <View key={index} style={styles.suggestionItem}>
                  <Text style={styles.suggestionBullet}>•</Text>
                  <Text style={styles.suggestionText}>{suggestion}</Text>
                </View>
              ))}
            </View>
          </Animated.View>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <Animated.View entering={FadeInDown.duration(800)} style={styles.header}>
          <Text style={styles.headerTitle}>Fitness Assessments</Text>
          <Text style={styles.headerSubtitle}>Choose your test and let AI analyze your performance</Text>
        </Animated.View>

        <Animated.View entering={FadeInDown.duration(800).delay(200)} style={styles.assessmentsGrid}>
          {assessmentTypes.map((assessment, index) => (
            <Animated.View 
              key={assessment.id} 
              entering={FadeInUp.duration(600).delay(100 * index)}
              style={styles.assessmentCard}
            >
              <TouchableOpacity onPress={() => startAssessment(assessment)}>
                <View style={styles.cardHeader}>
                  <Text style={styles.assessmentIcon}>{assessment.icon}</Text>
                  <View style={[styles.difficultyBadge, { backgroundColor: getDifficultyColor(assessment.difficulty) }]}>
                    <Text style={styles.difficultyText}>{assessment.difficulty}</Text>
                  </View>
                </View>
                
                <Text style={styles.assessmentName}>{assessment.name}</Text>
                <Text style={styles.assessmentDescription}>{assessment.description}</Text>
                
                <View style={styles.cardFooter}>
                  <View style={styles.durationContainer}>
                    <Timer size={14} color="#6B7280" />
                    <Text style={styles.durationText}>{assessment.duration}</Text>
                  </View>
                  <View style={[styles.startButton, { backgroundColor: assessment.color }]}>
                    <Camera size={16} color="#FFFFFF" />
                    <Text style={styles.startButtonText}>Start</Text>
                  </View>
                </View>
              </TouchableOpacity>
            </Animated.View>
          ))}
        </Animated.View>

        {/* Instructions */}
        <Animated.View entering={FadeInDown.duration(800).delay(1000)} style={styles.instructionsContainer}>
          <Text style={styles.instructionsTitle}>How it works</Text>
          <View style={styles.instructionsGrid}>
            <View style={styles.instructionStep}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>1</Text>
              </View>
              <Text style={styles.stepText}>Choose assessment</Text>
            </View>
            <View style={styles.instructionStep}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>2</Text>
              </View>
              <Text style={styles.stepText}>Record your performance</Text>
            </View>
            <View style={styles.instructionStep}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>3</Text>
              </View>
              <Text style={styles.stepText}>Get AI analysis & score</Text>
            </View>
          </View>
        </Animated.View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8FAFC',
  },
  scrollContent: {
    paddingBottom: 20,
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 20,
    backgroundColor: '#FFFFFF',
  },
  headerTitle: {
    fontFamily: 'Poppins-Bold',
    fontSize: 28,
    color: '#111827',
  },
  headerSubtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#6B7280',
    marginTop: 8,
  },
  assessmentsGrid: {
    paddingHorizontal: 20,
    paddingTop: 20,
    gap: 16,
  },
  assessmentCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 8,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  assessmentIcon: {
    fontSize: 32,
  },
  difficultyBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  difficultyText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 12,
    color: '#FFFFFF',
  },
  assessmentName: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 20,
    color: '#111827',
    marginBottom: 8,
  },
  assessmentDescription: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 16,
    lineHeight: 20,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  durationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  durationText: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#6B7280',
  },
  startButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 12,
    gap: 6,
  },
  startButtonText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
    color: '#FFFFFF',
  },
  instructionsContainer: {
    paddingHorizontal: 20,
    paddingTop: 32,
  },
  instructionsTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 20,
    color: '#111827',
    marginBottom: 20,
    textAlign: 'center',
  },
  instructionsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  instructionStep: {
    alignItems: 'center',
    flex: 1,
  },
  stepNumber: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#1E40AF',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  stepNumberText: {
    fontFamily: 'Poppins-Bold',
    fontSize: 16,
    color: '#FFFFFF',
  },
  stepText: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: '#6B7280',
    textAlign: 'center',
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  cameraHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
  },
  assessmentTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 20,
    color: '#FFFFFF',
  },
  flipButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraCenter: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordingIndicator: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
  },
  instructionText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 16,
    color: '#FFFFFF',
    textAlign: 'center',
  },
  durationText: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#E5E7EB',
    marginTop: 4,
  },
  cameraControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  cancelButton: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  cancelButtonText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 16,
    color: '#FFFFFF',
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#EF4444',
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordingButton: {
    backgroundColor: '#DC2626',
  },
  placeholder: {
    width: 80,
  },
  analyzingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F8FAFC',
    paddingHorizontal: 20,
  },
  analyzingContent: {
    alignItems: 'center',
  },
  analyzingTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 24,
    color: '#111827',
    marginTop: 20,
    marginBottom: 8,
  },
  analyzingSubtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#6B7280',
    textAlign: 'center',
    marginBottom: 32,
  },
  analyzingSteps: {
    gap: 12,
  },
  analyzingStep: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
  },
  resultsContainer: {
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  resultsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  resultsTitle: {
    fontFamily: 'Poppins-Bold',
    fontSize: 24,
    color: '#111827',
  },
  doneButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#1E40AF',
    borderRadius: 12,
  },
  doneButtonText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
    color: '#FFFFFF',
  },
  scoreCard: {
    padding: 24,
    borderRadius: 20,
    alignItems: 'center',
    marginBottom: 20,
  },
  scoreLabel: {
    fontFamily: 'Inter-Medium',
    fontSize: 16,
    color: '#E5E7EB',
  },
  scoreValue: {
    fontFamily: 'Poppins-Bold',
    fontSize: 48,
    color: '#FFFFFF',
    marginVertical: 8,
  },
  benchmarkText: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#FCD34D',
  },
  metricsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  metricCard: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  metricLabel: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 8,
  },
  metricValue: {
    fontFamily: 'Poppins-Bold',
    fontSize: 20,
    color: '#111827',
  },
  suggestionsContainer: {
    backgroundColor: '#FFFFFF',
    padding: 20,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  suggestionsTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 18,
    color: '#111827',
    marginBottom: 16,
  },
  suggestionItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  suggestionBullet: {
    fontFamily: 'Inter-Bold',
    fontSize: 16,
    color: '#1E40AF',
    marginRight: 12,
    marginTop: 2,
  },
  suggestionText: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#374151',
    lineHeight: 20,
    flex: 1,
  },
});