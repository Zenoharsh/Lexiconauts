import React, { useState } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChartBar as BarChart3, TrendingUp, Target, Clock, Trophy, Calendar, ArrowUp, ArrowDown } from 'lucide-react-native';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');

interface PerformanceMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  trend: number;
  icon: any;
  color: string;
}

interface WeeklyData {
  day: string;
  score: number;
}

export default function AnalyticsScreen() {
  const [selectedPeriod, setSelectedPeriod] = useState('week');

  const periods = ['Week', 'Month', 'Year'];
  
  const performanceMetrics: PerformanceMetric[] = [
    {
      id: '1',
      name: 'Overall Score',
      value: 87.4,
      unit: '/100',
      trend: 8.2,
      icon: Trophy,
      color: '#F59E0B'
    },
    {
      id: '2',
      name: 'Vertical Jump',
      value: 52.3,
      unit: 'cm',
      trend: 12.5,
      icon: TrendingUp,
      color: '#10B981'
    },
    {
      id: '3',
      name: 'Sprint Speed',
      value: 6.8,
      unit: 'sec',
      trend: -5.2,
      icon: Target,
      color: '#EF4444'
    },
    {
      id: '4',
      name: 'Endurance',
      value: 42.1,
      unit: 'min',
      trend: -8.7,
      icon: Clock,
      color: '#8B5CF6'
    }
  ];

  const weeklyData: WeeklyData[] = [
    { day: 'Mon', score: 82 },
    { day: 'Tue', score: 85 },
    { day: 'Wed', score: 79 },
    { day: 'Thu', score: 88 },
    { day: 'Fri', score: 91 },
    { day: 'Sat', score: 87 },
    { day: 'Sun', score: 89 }
  ];

  const maxScore = Math.max(...weeklyData.map(d => d.score));

  const strengths = ['Explosive Power', 'Coordination', 'Flexibility'];
  const improvements = ['Sprint Speed', 'Core Strength', 'Cardiovascular Endurance'];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <Animated.View entering={FadeInDown.duration(800)} style={styles.header}>
          <Text style={styles.headerTitle}>Performance Analytics</Text>
          <Text style={styles.headerSubtitle}>Track your progress and identify improvement areas</Text>
        </Animated.View>

        {/* Period Selector */}
        <Animated.View entering={FadeInDown.duration(800).delay(200)} style={styles.periodContainer}>
          <View style={styles.periodSelector}>
            {periods.map((period) => (
              <TouchableOpacity 
                key={period}
                style={[
                  styles.periodButton,
                  selectedPeriod === period.toLowerCase() && styles.activePeriod
                ]}
                onPress={() => setSelectedPeriod(period.toLowerCase())}
              >
                <Text style={[
                  styles.periodText,
                  selectedPeriod === period.toLowerCase() && styles.activePeriodText
                ]}>
                  {period}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </Animated.View>

        {/* Performance Metrics */}
        <Animated.View entering={FadeInDown.duration(800).delay(400)} style={styles.section}>
          <Text style={styles.sectionTitle}>Key Metrics</Text>
          <View style={styles.metricsGrid}>
            {performanceMetrics.map((metric, index) => {
              const IconComponent = metric.icon;
              return (
                <Animated.View 
                  key={metric.id}
                  entering={FadeInUp.duration(600).delay(100 * index)}
                  style={styles.metricCard}
                >
                  <View style={styles.metricHeader}>
                    <View style={[styles.metricIcon, { backgroundColor: metric.color + '20' }]}>
                      <IconComponent size={20} color={metric.color} />
                    </View>
                    <View style={[styles.trendIndicator, { 
                      backgroundColor: metric.trend > 0 ? '#10B981' : '#EF4444' 
                    }]}>
                      {metric.trend > 0 ? (
                        <ArrowUp size={12} color="#FFFFFF" />
                      ) : (
                        <ArrowDown size={12} color="#FFFFFF" />
                      )}
                      <Text style={styles.trendText}>
                        {Math.abs(metric.trend)}%
                      </Text>
                    </View>
                  </View>
                  <Text style={styles.metricValue}>
                    {metric.value}{metric.unit}
                  </Text>
                  <Text style={styles.metricName}>{metric.name}</Text>
                </Animated.View>
              );
            })}
          </View>
        </Animated.View>

        {/* Weekly Progress Chart */}
        <Animated.View entering={FadeInDown.duration(800).delay(600)} style={styles.section}>
          <Text style={styles.sectionTitle}>Weekly Progress</Text>
          <View style={styles.chartContainer}>
            <View style={styles.chartContent}>
              {weeklyData.map((data, index) => (
                <View key={data.day} style={styles.chartBar}>
                  <View 
                    style={[
                      styles.bar, 
                      { 
                        height: (data.score / maxScore) * 120,
                        backgroundColor: data.score === maxScore ? '#1E40AF' : '#E5E7EB'
                      }
                    ]} 
                  />
                  <Text style={styles.chartLabel}>{data.day}</Text>
                  <Text style={styles.chartValue}>{data.score}</Text>
                </View>
              ))}
            </View>
          </View>
        </Animated.View>

        {/* Strengths & Areas for Improvement */}
        <Animated.View entering={FadeInDown.duration(800).delay(800)} style={styles.section}>
          <View style={styles.insightsRow}>
            <View style={styles.insightCard}>
              <View style={styles.insightHeader}>
                <Trophy size={20} color="#10B981" />
                <Text style={styles.insightTitle}>Strengths</Text>
              </View>
              <View style={styles.insightsList}>
                {strengths.map((strength, index) => (
                  <View key={index} style={styles.insightItem}>
                    <Text style={styles.insightBullet}>•</Text>
                    <Text style={styles.insightText}>{strength}</Text>
                  </View>
                ))}
              </View>
            </View>

            <View style={styles.insightCard}>
              <View style={styles.insightHeader}>
                <Target size={20} color="#F97316" />
                <Text style={styles.insightTitle}>Focus Areas</Text>
              </View>
              <View style={styles.insightsList}>
                {improvements.map((improvement, index) => (
                  <View key={index} style={styles.insightItem}>
                    <Text style={styles.insightBullet}>•</Text>
                    <Text style={styles.insightText}>{improvement}</Text>
                  </View>
                ))}
              </View>
            </View>
          </View>
        </Animated.View>

        {/* AI Insights */}
        <Animated.View entering={FadeInDown.duration(800).delay(1000)} style={styles.section}>
          <Text style={styles.sectionTitle}>AI Insights</Text>
          <View style={styles.aiInsightCard}>
            <LinearGradient
              colors={['#8B5CF6', '#A855F7']}
              style={styles.aiInsightContent}
            >
              <Text style={styles.aiInsightTitle}>Performance Prediction</Text>
              <Text style={styles.aiInsightText}>
                Based on your current trajectory, you're likely to improve your overall score by 15-20% 
                over the next month with consistent training.
              </Text>
              <TouchableOpacity style={styles.aiInsightButton}>
                <Text style={styles.aiInsightButtonText}>View Training Plan</Text>
              </TouchableOpacity>
            </LinearGradient>
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
  periodContainer: {
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: '#FFFFFF',
  },
  periodSelector: {
    flexDirection: 'row',
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
    padding: 4,
  },
  periodButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  activePeriod: {
    backgroundColor: '#FFFFFF',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  periodText: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#6B7280',
  },
  activePeriodText: {
    color: '#111827',
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  sectionTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 20,
    color: '#111827',
    marginBottom: 16,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  metricCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    width: (width - 52) / 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  metricHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  metricIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  trendIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 2,
  },
  trendText: {
    fontFamily: 'Inter-Bold',
    fontSize: 10,
    color: '#FFFFFF',
  },
  metricValue: {
    fontFamily: 'Poppins-Bold',
    fontSize: 24,
    color: '#111827',
    marginBottom: 4,
  },
  metricName: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: '#6B7280',
  },
  chartContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  chartContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    height: 140,
    paddingBottom: 40,
  },
  chartBar: {
    alignItems: 'center',
    flex: 1,
  },
  bar: {
    width: 24,
    borderRadius: 12,
    marginBottom: 8,
  },
  chartLabel: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 4,
  },
  chartValue: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 10,
    color: '#111827',
  },
  insightsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  insightCard: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  insightHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  insightTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 16,
    color: '#111827',
  },
  insightsList: {
    gap: 8,
  },
  insightItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  insightBullet: {
    fontFamily: 'Inter-Bold',
    fontSize: 14,
    color: '#1E40AF',
    marginRight: 8,
    marginTop: 2,
  },
  insightText: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#374151',
    lineHeight: 16,
    flex: 1,
  },
  aiInsightCard: {
    borderRadius: 20,
    overflow: 'hidden',
  },
  aiInsightContent: {
    padding: 24,
  },
  aiInsightTitle: {
    fontFamily: 'Poppins-Bold',
    fontSize: 18,
    color: '#FFFFFF',
    marginBottom: 12,
  },
  aiInsightText: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#E5E7EB',
    lineHeight: 20,
    marginBottom: 20,
  },
  aiInsightButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  aiInsightButtonText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
    color: '#FFFFFF',
  },
});