import React, { useEffect, useState } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Dimensions, Image } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Trophy, Target, TrendingUp, Play } from 'lucide-react-native';
import Animated, { FadeInDown, FadeInUp, useSharedValue, withRepeat, withTiming, Easing, useAnimatedStyle } from 'react-native-reanimated';
import { AnimatedGlowIcon, FireProgress } from '../../components/AnimatedGlowIcon';
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useNavigation } from "@react-navigation/native";
import { useRouter } from "expo-router";
import { Button } from "react-native";


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
  const router = useRouter();
  const navigation = useNavigation();


  const [achievements] = useState<Achievement[]>([
    {
      id: '1',
      title: 'Speed Demon',
      description: 'Complete 5 sprint tests',
      progress: 3,
      total: 5,
      icon: '🏃‍♂️',
      color: '#F97316',
    },
    {
      id: '2',
      title: 'Jump Master',
      description: 'Achieve 60cm vertical jump',
      progress: 45,
      total: 60,
      icon: '🦘',
      color: '#10B981',
    },
    {
      id: '3',
      title: 'Endurance Pro',
      description: 'Complete 10km run under 45 min',
      progress: 42,
      total: 45,
      icon: '💪',
      color: '#8B5CF6',
    },
  ]);

  const [recentActivities] = useState<RecentActivity[]>([
    {
      id: '1',
      type: 'Vertical Jump',
      title: 'Personal Best!',
      score: 52.3,
      date: '2 hours ago',
      improvement: 8.2,
    },
    {
      id: '2',
      type: 'Sprint Test',
      title: 'Great Performance',
      score: 6.8,
      date: 'Yesterday',
      improvement: -0.3,
    },
    {
      id: '3',
      type: 'Sit-ups',
      title: 'Steady Progress',
      score: 48,
      date: '2 days ago',
      improvement: 4,
    },
  ]);
 const logout = async () => {
    try {
      await AsyncStorage.removeItem("token"); // token clear
      
      navigation.reset({
        index: 0,
        routes: [{ name: "Login" }], // 👈 Login screen pe redirect
      });
    } catch (error) {
      console.log("Logout error:", error);
    }
  };
  const userName = 'Arjun Singh';
  const overallScore = 87;
  const rank = 23;
  const totalAthletes = 1247;

  // Small floating animation for emojis on Achievements
  const float = useSharedValue(0);
  useEffect(() => {
    float.value = withRepeat(withTiming(1, { duration: 1600, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, [float]);
  const floatStyle = useAnimatedStyle(() => ({ transform: [{ translateY: float.value * -2 }] }));

  return (
    <SafeAreaView style={styles.container}>
      {/* {🔥 Logout Button top-right} */}
    <TouchableOpacity
      onPress={logout}
      style={{ position: "absolute", top: 40, right: 20 }}
    >
      <Text style={{ color: "red", fontWeight: "bold", fontSize: 16 }}>
        Logout
      </Text>
    </TouchableOpacity>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
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

        <Animated.View entering={FadeInDown.duration(800).delay(200)}>
          <LinearGradient colors={['#6D28D9', '#0B1220']} style={styles.performanceCard}>
            <View style={styles.performanceHeader}>
              <View>
                <Text style={styles.performanceTitle}>Overall Performance</Text>
                <Text style={styles.performanceSubtitle}>Based on last 30 days</Text>
              </View>
              <AnimatedGlowIcon size={44} glowColors={['#F59E0B', '#FDE047']} bgColors={['#7C3AED', '#0B1220']}>
                <Trophy size={26} color="#FFFFFF" />
              </AnimatedGlowIcon>
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

        <Animated.View entering={FadeInDown.duration(800).delay(400)} style={styles.section}>
          <Text style={styles.sectionTitle}>Quick Start Assessment</Text>
          <View style={styles.quickActionsGrid}>
            <LinearGradient
              colors={['#0B1220', '#3B0764']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={[styles.quickActionCard]}
            >
              <AnimatedGlowIcon size={52} glowColors={['#FF8A00', '#FFD700']} bgColors={['#1E1B4B', '#6D28D9']}>
                <Play size={26} color="#FFFFFF" />
              </AnimatedGlowIcon>
              <Text style={styles.actionTitleDark}>Video Test</Text>
              <Text style={styles.actionSubtitleDark}>Record & analyze</Text>
            </LinearGradient>

            <LinearGradient
              colors={['#0B1220', '#1F2937']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={[styles.quickActionCard]}
            >
              <AnimatedGlowIcon size={52} glowColors={['#A78BFA', '#FDE047']} bgColors={['#0F172A', '#7C3AED']}>
                <Target size={26} color="#FFFFFF" />
              </AnimatedGlowIcon>
              <Text style={styles.actionTitleDark}>Live Assessment</Text>
              <Text style={styles.actionSubtitleDark}>Real-time scoring</Text>
            </LinearGradient>
          </View>
        </Animated.View>

        <Animated.View entering={FadeInDown.duration(800).delay(600)} style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Achievements</Text>
            <TouchableOpacity>
              <Text style={styles.viewAllText}>View All</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.achievementsContainer}>
            {achievements.map((achievement, index) => {
              const pct = (achievement.progress / achievement.total) * 100;
              return (
                <Animated.View key={achievement.id} entering={FadeInUp.duration(600).delay(100 * index)} style={styles.achievementCardDark}>
                  <View style={styles.achievementHeader}>
                    <Animated.Text style={[styles.achievementIcon, floatStyle]}>{achievement.icon}</Animated.Text>
                    <Text style={styles.achievementProgressDark}>
                      {achievement.progress}/{achievement.total}
                    </Text>
                  </View>
                  <Text style={styles.achievementTitleDark}>{achievement.title}</Text>
                  <Text style={styles.achievementDescriptionDark}>{achievement.description}</Text>
                  <FireProgress progress={pct} />
                </Animated.View>
              );
            })}
          </ScrollView>
        </Animated.View>

        <Animated.View entering={FadeInDown.duration(800).delay(800)} style={styles.section}>
          <Text style={styles.sectionTitle}>Recent Activity</Text>
          <View style={styles.activitiesContainer}>
            {recentActivities.map((activity, index) => (
              <Animated.View key={activity.id} entering={FadeInUp.duration(600).delay(100 * index)} style={styles.activityCard}>
                <View style={styles.activityLeft}>
                  <AnimatedGlowIcon size={40} glowColors={activity.improvement > 0 ? ['#10B981', '#6EE7B7'] : ['#EF4444', '#FCA5A5']} bgColors={['#0B1220', '#1F2937']}>
                    <TrendingUp size={16} color="#FFFFFF" />
                  </AnimatedGlowIcon>
                  <View style={styles.activityInfo}>
                    <Text style={styles.activityTypeDark}>{activity.type}</Text>
                    <Text style={styles.activityTitleDark}>{activity.title}</Text>
                    <Text style={styles.activityDateDark}>{activity.date}</Text>
                  </View>
                </View>
                <View style={styles.activityRight}>
                  <Text style={styles.activityScoreDark}>{activity.score}</Text>
                  <Text style={[styles.activityImprovement, { color: activity.improvement > 0 ? '#10B981' : '#EF4444' }]}>
                    {activity.improvement > 0 ? '+' : ''}
                    {activity.improvement}%
                  </Text>
                </View>
              </Animated.View>
            ))}
          </View>
        </Animated.View>

        <Animated.View entering={FadeInDown.duration(800).delay(1000)} style={styles.section}>
          <Text style={styles.sectionTitle}>Upcoming Events</Text>
          <LinearGradient colors={['#0B1220', '#1F2937']} style={styles.eventCardDark}>
            <View style={styles.eventHeader}>
              <AnimatedGlowIcon size={40} glowColors={['#A78BFA', '#FDE047']} bgColors={['#0B1220', '#6D28D9']}>
                <View />
              </AnimatedGlowIcon>
              <View style={styles.eventInfo}> 
                <Text style={styles.eventTitleDark}>Regional Assessment</Text>
                <View style={styles.eventLocation}> 
                  <Text style={styles.eventLocationTextDark}>Mumbai Sports Complex</Text>
                </View>
              </View>
            </View>
            <Text style={styles.eventDateDark}>March 15, 2025</Text>
          </LinearGradient>
        </Animated.View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0C14',
  },
  scrollContent: {
    paddingBottom: 20,
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: 'transparent',
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  greeting: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#A5B4FC',
  },
  userName: {
    fontFamily: 'Poppins-Bold',
    fontSize: 24,
    color: '#FFFFFF',
    marginTop: 4,
  },
  avatarContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#111827',
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
    color: '#CBD5E1',
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
    backgroundColor: 'rgba(255,255,255,0.15)',
    marginHorizontal: 16,
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  sectionTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 20,
    color: '#FFFFFF',
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
    color: '#A78BFA',
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
  actionTitleDark: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 16,
    color: '#FFFFFF',
    marginBottom: 4,
    marginTop: 8,
  },
  actionSubtitleDark: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#CBD5E1',
    textAlign: 'center',
  },
  achievementsContainer: {
    marginVertical: 8,
  },
  achievementCardDark: {
    width: 160,
    backgroundColor: '#0B1220',
    borderRadius: 16,
    padding: 16,
    marginRight: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 10,
    elevation: 6,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.25)'
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
  achievementProgressDark: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 12,
    color: '#E5E7EB',
  },
  achievementTitleDark: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 14,
    color: '#FFFFFF',
    marginBottom: 4,
  },
  achievementDescriptionDark: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#CBD5E1',
    marginBottom: 12,
  },
  progressBarContainer: {
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressDynamic: {
    height: '100%',
    borderRadius: 3,
    overflow: 'hidden',
    position: 'relative',
  },
  activitiesContainer: {
    gap: 12,
  },
  activityCard: {
    backgroundColor: '#0B1220',
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 10,
    elevation: 6,
    borderWidth: 1,
    borderColor: 'rgba(99,102,241,0.18)'
  },
  activityLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  activityInfo: {
    flex: 1,
    marginLeft: 8,
  },
  activityTypeDark: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#FFFFFF',
  },
  activityTitleDark: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#CBD5E1',
    marginTop: 2,
  },
  activityDateDark: {
    fontFamily: 'Inter-Regular',
    fontSize: 11,
    color: '#94A3B8',
    marginTop: 2,
  },
  activityRight: {
    alignItems: 'flex-end',
  },
  activityScoreDark: {
    fontFamily: 'Poppins-Bold',
    fontSize: 16,
    color: '#FFFFFF',
  },
  activityImprovement: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    marginTop: 2,
  },
  eventCardDark: {
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 10,
    elevation: 6,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.25)'
  },
  eventHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  eventInfo: {
    flex: 1,
    marginLeft: 12,
  },
  eventTitleDark: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 16,
    color: '#FFFFFF',
  },
  eventLocation: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  eventLocationTextDark: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#CBD5E1',
    marginLeft: 4,
  },
  eventDateDark: {
    fontFamily: 'Poppins-Medium',
    fontSize: 14,
    color: '#A78BFA',
  },
});
