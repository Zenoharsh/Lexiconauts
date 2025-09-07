import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Dimensions, Image } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Trophy, Target, TrendingUp, Award, Play, Calendar, MapPin } from 'lucide-react-native';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';

const { width } = Dimensions.get('window');

interface Achievement {
  id: string;
  title: string;
  description: string;
  progress: number;
  total: number;
  icon: string;
  color: string;
}

interface RecentActivity {
  id: string;
  type: string;
  title: string;
  score: number;
  date: string;
  improvement: number;
}

export default function HomeScreen() {
  const [achievements, setAchievements] = useState<Achievement[]>([
    {
      id: '1',
      title: 'Speed Demon',
      description: 'Complete 5 sprint tests',
      progress: 3,
      total: 5,
      icon: '🏃‍♂️',
      color: '#F97316'
    },
    {
      id: '2',
      title: 'Jump Master',
      description: 'Achieve 60cm vertical jump',
      progress: 45,
      total: 60,
      icon: '🦘',
      color: '#10B981'
    },
    {
      id: '3',
      title: 'Endurance Pro',
      description: 'Complete 10km run under 45 min',
      progress: 42,
      total: 45,
      icon: '💪',
      color: '#8B5CF6'
    }
  ]);

  const [recentActivities, setRecentActivities] = useState<RecentActivity[]>([
    {
      id: '1',
      type: 'Vertical Jump',
      title: 'Personal Best!',
      score: 52.3,
      date: '2 hours ago',
      improvement: 8.2
    },
    {
      id: '2',
      type: 'Sprint Test',
      title: 'Great Performance',
      score: 6.8,
      date: 'Yesterday',
      improvement: -0.3
    },
    {
      id: '3',
      type: 'Sit-ups',
      title: 'Steady Progress',
      score: 48,
      date: '2 days ago',
      improvement: 4
    }
  ]);

  const userName = "Arjun Singh";
  const overallScore = 87;
  const rank = 23;
  const totalAthletes = 1247;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <Animated.View entering={FadeInDown.duration(800)} style={styles.header}>
          <View style={styles.headerContent}>
            <View>
              <Text style={styles.greeting}>Good morning,</Text>
              <Text style={styles.userName}>{userName}</Text>
            </View>
            <TouchableOpacity style={styles.avatarContainer}>
              <Image 
                source={{ uri: 'https://images.pexels.com/photos/1043471/pexels-photo-1043471.jpeg?auto=compress&cs=tinysrgb&w=100&h=100&dpr=2' }}
                style={styles.avatar}
              />
            </TouchableOpacity>
          </View>
        </Animated.View>

        {/* Performance Overview Card */}
        <Animated.View entering={FadeInDown.duration(800).delay(200)}>
          <LinearGradient
            colors={['#1E40AF', '#3B82F6']}
            style={styles.performanceCard}
          >
            <View style={styles.performanceHeader}>
              <View>
                <Text style={styles.performanceTitle}>Overall Performance</Text>
                <Text style={styles.performanceSubtitle}>Based on last 30 days</Text>
              </View>
              <Trophy size={32} color="#FCD34D" />
            </View>
            <View style={styles.performanceStats}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{overallScore}</Text>
                <Text style={styles.statLabel}>Score</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statValue}>#{rank}</Text>
                <Text style={styles.statLabel}>Rank</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{totalAthletes}</Text>
                <Text style={styles.statLabel}>Athletes</Text>
              </View>
            </View>
          </LinearGradient>
        </Animated.View>

        {/* Quick Actions */}
        <Animated.View entering={FadeInDown.duration(800).delay(400)} style={styles.section}>
          <Text style={styles.sectionTitle}>Quick Start Assessment</Text>
          <View style={styles.quickActionsGrid}>
            <TouchableOpacity style={[styles.quickActionCard, { backgroundColor: '#FEF3F2' }]}>
              <View style={[styles.actionIconContainer, { backgroundColor: '#F97316' }]}>
                <Play size={24} color="#FFFFFF" />
              </View>
              <Text style={styles.actionTitle}>Video Test</Text>
              <Text style={styles.actionSubtitle}>Record & analyze</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={[styles.quickActionCard, { backgroundColor: '#F0FDF4' }]}>
              <View style={[styles.actionIconContainer, { backgroundColor: '#10B981' }]}>
                <Target size={24} color="#FFFFFF" />
              </View>
              <Text style={styles.actionTitle}>Live Assessment</Text>
              <Text style={styles.actionSubtitle}>Real-time scoring</Text>
            </TouchableOpacity>
          </View>
        </Animated.View>

        {/* Achievements */}
        <Animated.View entering={FadeInDown.duration(800).delay(600)} style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Achievements</Text>
            <TouchableOpacity>
              <Text style={styles.viewAllText}>View All</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.achievementsContainer}>
            {achievements.map((achievement, index) => (
              <Animated.View 
                key={achievement.id} 
                entering={FadeInUp.duration(600).delay(100 * index)}
                style={styles.achievementCard}
              >
                <View style={styles.achievementHeader}>
                  <Text style={styles.achievementIcon}>{achievement.icon}</Text>
                  <Text style={styles.achievementProgress}>
                    {achievement.progress}/{achievement.total}
                  </Text>
                </View>
                <Text style={styles.achievementTitle}>{achievement.title}</Text>
                <Text style={styles.achievementDescription}>{achievement.description}</Text>
                <View style={styles.progressBarContainer}>
                  <View style={[styles.progressBarFill, { 
                    width: `${(achievement.progress / achievement.total) * 100}%`,
                    backgroundColor: achievement.color 
                  }]} />
                </View>
              </Animated.View>
            ))}
          </ScrollView>
        </Animated.View>

        {/* Recent Activity */}
        <Animated.View entering={FadeInDown.duration(800).delay(800)} style={styles.section}>
          <Text style={styles.sectionTitle}>Recent Activity</Text>
          <View style={styles.activitiesContainer}>
            {recentActivities.map((activity, index) => (
              <Animated.View 
                key={activity.id} 
                entering={FadeInUp.duration(600).delay(100 * index)}
                style={styles.activityCard}
              >
                <View style={styles.activityLeft}>
                  <View style={[styles.activityIconContainer, { 
                    backgroundColor: activity.improvement > 0 ? '#10B981' : '#EF4444' 
                  }]}>
                    <TrendingUp size={16} color="#FFFFFF" />
                  </View>
                  <View style={styles.activityInfo}>
                    <Text style={styles.activityType}>{activity.type}</Text>
                    <Text style={styles.activityTitle}>{activity.title}</Text>
                    <Text style={styles.activityDate}>{activity.date}</Text>
                  </View>
                </View>
                <View style={styles.activityRight}>
                  <Text style={styles.activityScore}>{activity.score}</Text>
                  <Text style={[styles.activityImprovement, {
                    color: activity.improvement > 0 ? '#10B981' : '#EF4444'
                  }]}>
                    {activity.improvement > 0 ? '+' : ''}{activity.improvement}%
                  </Text>
                </View>
              </Animated.View>
            ))}
          </View>
        </Animated.View>

        {/* Upcoming Events */}
        <Animated.View entering={FadeInDown.duration(800).delay(1000)} style={styles.section}>
          <Text style={styles.sectionTitle}>Upcoming Events</Text>
          <View style={styles.eventCard}>
            <View style={styles.eventHeader}>
              <View style={styles.eventIconContainer}>
                <Calendar size={20} color="#1E40AF" />
              </View>
              <View style={styles.eventInfo}>
                <Text style={styles.eventTitle}>Regional Assessment</Text>
                <View style={styles.eventLocation}>
                  <MapPin size={14} color="#6B7280" />
                  <Text style={styles.eventLocationText}>Mumbai Sports Complex</Text>
                </View>
              </View>
            </View>
            <Text style={styles.eventDate}>March 15, 2025</Text>
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
    paddingVertical: 16,
    backgroundColor: '#FFFFFF',
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  greeting: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#6B7280',
  },
  userName: {
    fontFamily: 'Poppins-Bold',
    fontSize: 24,
    color: '#111827',
    marginTop: 4,
  },
  avatarContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#E5E7EB',
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
  },
  performanceCard: {
    marginHorizontal: 20,
    marginVertical: 20,
    borderRadius: 16,
    padding: 20,
  },
  performanceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  performanceTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 18,
    color: '#FFFFFF',
  },
  performanceSubtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#E5E7EB',
    marginTop: 4,
  },
  performanceStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statValue: {
    fontFamily: 'Poppins-Bold',
    fontSize: 24,
    color: '#FFFFFF',
  },
  statLabel: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: '#E5E7EB',
    marginTop: 4,
  },
  statDivider: {
    width: 1,
    height: 40,
    backgroundColor: '#3B82F6',
    marginHorizontal: 16,
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
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  viewAllText: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#1E40AF',
  },
  quickActionsGrid: {
    flexDirection: 'row',
    gap: 12,
  },
  quickActionCard: {
    flex: 1,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  actionIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  actionTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 16,
    color: '#111827',
    marginBottom: 4,
  },
  actionSubtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#6B7280',
    textAlign: 'center',
  },
  achievementsContainer: {
    marginVertical: 8,
  },
  achievementCard: {
    width: 160,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    marginRight: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  achievementHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  achievementIcon: {
    fontSize: 24,
  },
  achievementProgress: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 12,
    color: '#6B7280',
  },
  achievementTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 14,
    color: '#111827',
    marginBottom: 4,
  },
  achievementDescription: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 12,
  },
  progressBarContainer: {
    height: 6,
    backgroundColor: '#E5E7EB',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 3,
  },
  activitiesContainer: {
    gap: 12,
  },
  activityCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  activityLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  activityIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  activityInfo: {
    flex: 1,
  },
  activityType: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#111827',
  },
  activityTitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#6B7280',
    marginTop: 2,
  },
  activityDate: {
    fontFamily: 'Inter-Regular',
    fontSize: 11,
    color: '#9CA3AF',
    marginTop: 2,
  },
  activityRight: {
    alignItems: 'flex-end',
  },
  activityScore: {
    fontFamily: 'Poppins-Bold',
    fontSize: 16,
    color: '#111827',
  },
  activityImprovement: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    marginTop: 2,
  },
  eventCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  eventHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  eventIconContainer: {
    width: 40,
    height: 40,
    backgroundColor: '#EBF2FF',
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  eventInfo: {
    flex: 1,
  },
  eventTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 16,
    color: '#111827',
  },
  eventLocation: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  eventLocationText: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#6B7280',
    marginLeft: 4,
  },
  eventDate: {
    fontFamily: 'Poppins-Medium',
    fontSize: 14,
    color: '#1E40AF',
  },
});