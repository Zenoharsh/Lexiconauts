import React, { useState } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Image, Switch } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { User, Settings, Trophy, Target, Calendar, MapPin, CreditCard as Edit3, Bell, Shield, CircleHelp as HelpCircle, LogOut } from 'lucide-react-native';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';

interface ProfileStat {
  label: string;
  value: string;
}

interface Achievement {
  id: string;
  title: string;
  icon: string;
  date: string;
}

export default function ProfileScreen() {
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [biometricsEnabled, setBiometricsEnabled] = useState(false);

  const userProfile = {
    name: 'Arjun Singh',
    email: 'arjun.singh@gmail.com',
    avatar: 'https://images.pexels.com/photos/1043471/pexels-photo-1043471.jpeg?auto=compress&cs=tinysrgb&w=200&h=200&dpr=2',
    sport: 'Football',
    position: 'Midfielder',
    location: 'Mumbai, Maharashtra',
    joinDate: 'January 2024',
    overallScore: 87.4,
    rank: 23,
    assessmentsCompleted: 47
  };

  const profileStats: ProfileStat[] = [
    { label: 'Assessments', value: '47' },
    { label: 'Avg Score', value: '87.4' },
    { label: 'Best Rank', value: '#18' },
    { label: 'Streak', value: '12 days' }
  ];

  const recentAchievements: Achievement[] = [
    { id: '1', title: 'Speed Demon', icon: '🏃‍♂️', date: '2 days ago' },
    { id: '2', title: 'Consistency King', icon: '👑', date: '1 week ago' },
    { id: '3', title: 'Jump Master', icon: '🦘', date: '2 weeks ago' }
  ];

  const menuItems = [
    { icon: Bell, title: 'Notifications', hasSwitch: true, value: notificationsEnabled, onToggle: setNotificationsEnabled },
    { icon: Shield, title: 'Biometric Login', hasSwitch: true, value: biometricsEnabled, onToggle: setBiometricsEnabled },
    { icon: Target, title: 'Training Goals', hasSwitch: false },
    { icon: Calendar, title: 'Assessment History', hasSwitch: false },
    { icon: HelpCircle, title: 'Help & Support', hasSwitch: false },
    { icon: Settings, title: 'App Settings', hasSwitch: false },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Profile Header */}
        <Animated.View entering={FadeInDown.duration(800)} style={styles.profileHeader}>
          <LinearGradient
            colors={['#6D28D9', '#0B1220']}
            style={styles.profileGradient}
          >
            <View style={styles.profileInfo}>
              <Image source={{ uri: userProfile.avatar }} style={styles.profileAvatar} />
              <TouchableOpacity style={styles.editButton}>
                <Edit3 size={16} color="#FFFFFF" />
              </TouchableOpacity>
            </View>
            <Text style={styles.profileName}>{userProfile.name}</Text>
            <Text style={styles.profileEmail}>{userProfile.email}</Text>
            
            <View style={styles.profileDetails}>
              <View style={styles.profileDetailItem}>
                <Target size={14} color="#E5E7EB" />
                <Text style={styles.profileDetailText}>{userProfile.sport}</Text>
              </View>
              <View style={styles.profileDetailItem}>
                <MapPin size={14} color="#E5E7EB" />
                <Text style={styles.profileDetailText}>{userProfile.location}</Text>
              </View>
            </View>
          </LinearGradient>
        </Animated.View>

        {/* Stats Grid */}
        <Animated.View entering={FadeInDown.duration(800).delay(200)} style={styles.statsContainer}>
          <View style={styles.statsGrid}>
            {profileStats.map((stat, index) => (
              <Animated.View 
                key={stat.label}
                entering={FadeInUp.duration(600).delay(100 * index)}
                style={styles.statCard}
              >
                <Text style={styles.statValue}>{stat.value}</Text>
                <Text style={styles.statLabel}>{stat.label}</Text>
              </Animated.View>
            ))}
          </View>
        </Animated.View>

        {/* Recent Achievements */}
        <Animated.View entering={FadeInDown.duration(800).delay(400)} style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Recent Achievements</Text>
            <TouchableOpacity>
              <Text style={styles.viewAllText}>View All</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.achievementsScroll}>
            {recentAchievements.map((achievement, index) => (
              <Animated.View 
                key={achievement.id}
                entering={FadeInUp.duration(600).delay(100 * index)}
                style={styles.achievementCard}
              >
                <Text style={styles.achievementIcon}>{achievement.icon}</Text>
                <Text style={styles.achievementTitle}>{achievement.title}</Text>
                <Text style={styles.achievementDate}>{achievement.date}</Text>
              </Animated.View>
            ))}
          </ScrollView>
        </Animated.View>

        {/* Menu Items */}
        <Animated.View entering={FadeInDown.duration(800).delay(600)} style={styles.section}>
          <Text style={styles.sectionTitle}>Settings</Text>
          <View style={styles.menuContainer}>
            {menuItems.map((item, index) => {
              const IconComponent = item.icon;
              return (
                <Animated.View 
                  key={item.title}
                  entering={FadeInUp.duration(600).delay(50 * index)}
                >
                  <TouchableOpacity style={styles.menuItem}>
                    <View style={styles.menuItemLeft}>
                      <View style={styles.menuIcon}>
                        <IconComponent size={20} color="#6B7280" />
                      </View>
                      <Text style={styles.menuItemText}>{item.title}</Text>
                    </View>
                    {item.hasSwitch && item.onToggle ? (
                      <Switch
                        value={item.value}
                        onValueChange={item.onToggle}
                        trackColor={{ false: '#E5E7EB', true: '#1E40AF' }}
                        thumbColor="#FFFFFF"
                      />
                    ) : (
                      <Text style={styles.menuArrow}>›</Text>
                    )}
                  </TouchableOpacity>
                </Animated.View>
              );
            })}
          </View>
        </Animated.View>

        {/* Logout Button */}
        <Animated.View entering={FadeInDown.duration(800).delay(800)} style={styles.section}>
          <TouchableOpacity style={styles.logoutButton}>
            <LogOut size={20} color="#EF4444" />
            <Text style={styles.logoutText}>Sign Out</Text>
          </TouchableOpacity>
        </Animated.View>

        <View style={styles.bottomSpacing} />
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
  profileHeader: {
    marginBottom: 20,
  },
  profileGradient: {
    paddingHorizontal: 20,
    paddingTop: 40,
    paddingBottom: 32,
    alignItems: 'center',
  },
  profileInfo: {
    position: 'relative',
    marginBottom: 16,
  },
  profileAvatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderWidth: 4,
    borderColor: '#FFFFFF',
  },
  editButton: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#F97316',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#FFFFFF',
  },
  profileName: {
    fontFamily: 'Poppins-Bold',
    fontSize: 24,
    color: '#FFFFFF',
    marginBottom: 4,
  },
  profileEmail: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#E5E7EB',
    marginBottom: 16,
  },
  profileDetails: {
    flexDirection: 'row',
    gap: 20,
  },
  profileDetailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  profileDetailText: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#E5E7EB',
  },
  statsContainer: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 12,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#0B1220',
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 4,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.25)'
  },
  statValue: {
    fontFamily: 'Poppins-Bold',
    fontSize: 20,
    color: '#FFFFFF',
  },
  statLabel: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: '#CBD5E1',
    marginTop: 4,
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 20,
    color: '#FFFFFF',
  },
  viewAllText: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#1E40AF',
  },
  achievementsScroll: {
    marginHorizontal: -4,
  },
  achievementCard: {
    width: 100,
    backgroundColor: '#0B1220',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    marginHorizontal: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 4,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.25)'
  },
  achievementIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  achievementTitle: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 12,
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 4,
  },
  achievementDate: {
    fontFamily: 'Inter-Regular',
    fontSize: 10,
    color: '#94A3B8',
    textAlign: 'center',
  },
  menuContainer: {
    backgroundColor: '#0B1220',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 4,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.25)'
  },
  menuItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(167,139,250,0.2)',
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  menuIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#1E1B4B',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  menuItemText: {
    fontFamily: 'Inter-Medium',
    fontSize: 16,
    color: '#FFFFFF',
  },
  menuArrow: {
    fontFamily: 'Inter-Regular',
    fontSize: 20,
    color: '#CBD5E1',
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0B1220',
    paddingVertical: 16,
    borderRadius: 16,
    gap: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 4,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.25)'
  },
  logoutText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 16,
    color: '#EF4444',
  },
  bottomSpacing: {
    height: 20,
  },
});
