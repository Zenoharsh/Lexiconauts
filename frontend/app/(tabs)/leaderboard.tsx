import React, { useState } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Trophy, Medal, Award, Filter, TrendingUp, Crown } from 'lucide-react-native';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';

interface Athlete {
  id: string;
  name: string;
  avatar: string;
  score: number;
  region: string;
  ageGroup: string;
  sport: string;
  improvement: number;
  rank: number;
}

interface LeaderboardCategory {
  id: string;
  name: string;
  icon: any;
  color: string;
}

export default function LeaderboardScreen() {
  const [selectedCategory, setSelectedCategory] = useState('overall');
  const [selectedFilter, setSelectedFilter] = useState('all');

  const categories: LeaderboardCategory[] = [
    { id: 'overall', name: 'Overall', icon: Trophy, color: '#F59E0B' },
    { id: 'sprint', name: 'Sprint', icon: TrendingUp, color: '#EF4444' },
    { id: 'jump', name: 'Jump', icon: Award, color: '#10B981' },
    { id: 'endurance', name: 'Endurance', icon: Medal, color: '#8B5CF6' },
  ];

  const filters = ['All', 'My Age', 'My Region'];

  const athletes: Athlete[] = [
    {
      id: '1',
      name: 'Priya Sharma',
      avatar: 'https://images.pexels.com/photos/1130626/pexels-photo-1130626.jpeg?auto=compress&cs=tinysrgb&w=100&h=100&dpr=2',
      score: 94.2,
      region: 'Maharashtra',
      ageGroup: '18-22',
      sport: 'Track & Field',
      improvement: 12.5,
      rank: 1
    },
    {
      id: '2',
      name: 'Rahul Verma',
      avatar: 'https://images.pexels.com/photos/1043471/pexels-photo-1043471.jpeg?auto=compress&cs=tinysrgb&w=100&h=100&dpr=2',
      score: 91.8,
      region: 'Punjab',
      ageGroup: '18-22',
      sport: 'Wrestling',
      improvement: 8.3,
      rank: 2
    },
    {
      id: '3',
      name: 'Anita Patel',
      avatar: 'https://images.pexels.com/photos/1536619/pexels-photo-1536619.jpeg?auto=compress&cs=tinysrgb&w=100&h=100&dpr=2',
      score: 89.6,
      region: 'Gujarat',
      ageGroup: '18-22',
      sport: 'Basketball',
      improvement: 15.2,
      rank: 3
    },
    {
      id: '4',
      name: 'Arjun Singh',
      avatar: 'https://images.pexels.com/photos/1270076/pexels-photo-1270076.jpeg?auto=compress&cs=tinysrgb&w=100&h=100&dpr=2',
      score: 87.4,
      region: 'Rajasthan',
      ageGroup: '18-22',
      sport: 'Football',
      improvement: 6.7,
      rank: 4
    },
    {
      id: '5',
      name: 'Meera Kumar',
      avatar: 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=100&h=100&dpr=2',
      score: 85.9,
      region: 'Kerala',
      ageGroup: '18-22',
      sport: 'Swimming',
      improvement: 4.1,
      rank: 5
    }
  ];

  const getRankIcon = (rank: number) => {
    if (rank === 1) return <Crown size={20} color="#F59E0B" />;
    if (rank === 2) return <Medal size={20} color="#9CA3AF" />;
    if (rank === 3) return <Award size={20} color="#CD7C2F" />;
    return null;
  };

  const getRankColors = (rank: number) => {
    if (rank === 1) return ['#F59E0B', '#FCD34D'];
    if (rank === 2) return ['#9CA3AF', '#D1D5DB'];
    if (rank === 3) return ['#CD7C2F', '#FBBF24'];
    return ['#E5E7EB', '#F3F4F6'];
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <Animated.View entering={FadeInDown.duration(800)} style={styles.header}>
          <Text style={styles.headerTitle}>Leaderboard</Text>
          <Text style={styles.headerSubtitle}>National talent rankings</Text>
        </Animated.View>

        {/* Categories */}
        <Animated.View entering={FadeInDown.duration(800).delay(200)} style={styles.categoriesContainer}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.categoriesScroll}>
            {categories.map((category, index) => {
              const IconComponent = category.icon;
              const isSelected = selectedCategory === category.id;
              return (
                <TouchableOpacity 
                  key={category.id}
                  style={[
                    styles.categoryCard,
                    isSelected && { backgroundColor: category.color }
                  ]}
                  onPress={() => setSelectedCategory(category.id)}
                >
                  <IconComponent 
                    size={20} 
                    color={isSelected ? '#FFFFFF' : category.color} 
                  />
                  <Text style={[
                    styles.categoryText,
                    { color: isSelected ? '#FFFFFF' : category.color }
                  ]}>
                    {category.name}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        </Animated.View>

        {/* Filters */}
        <Animated.View entering={FadeInDown.duration(800).delay(400)} style={styles.filtersContainer}>
          <View style={styles.filtersRow}>
            {filters.map((filter, index) => (
              <TouchableOpacity 
                key={filter}
                style={[
                  styles.filterButton,
                  selectedFilter === filter.toLowerCase().replace(' ', '') && styles.activeFilter
                ]}
                onPress={() => setSelectedFilter(filter.toLowerCase().replace(' ', ''))}
              >
                <Text style={[
                  styles.filterText,
                  selectedFilter === filter.toLowerCase().replace(' ', '') && styles.activeFilterText
                ]}>
                  {filter}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </Animated.View>

        {/* Top 3 Podium */}
        <Animated.View entering={FadeInUp.duration(800).delay(600)} style={styles.podiumContainer}>
          <View style={styles.podium}>
            {/* 2nd Place */}
            <View style={styles.podiumPosition}>
              <LinearGradient
                colors={getRankColors(2)}
                style={[styles.podiumCard, styles.secondPlace]}
              >
                <Image source={{ uri: athletes[1].avatar }} style={styles.podiumAvatar} />
                <View style={styles.podiumRank}>
                  {getRankIcon(2)}
                </View>
              </LinearGradient>
              <Text style={styles.podiumName}>{athletes[1].name.split(' ')[0]}</Text>
              <Text style={styles.podiumScore}>{athletes[1].score}</Text>
            </View>

            {/* 1st Place */}
            <View style={styles.podiumPosition}>
              <LinearGradient
                colors={getRankColors(1)}
                style={[styles.podiumCard, styles.firstPlace]}
              >
                <Image source={{ uri: athletes[0].avatar }} style={styles.podiumAvatar} />
                <View style={styles.podiumRank}>
                  {getRankIcon(1)}
                </View>
              </LinearGradient>
              <Text style={styles.podiumName}>{athletes[0].name.split(' ')[0]}</Text>
              <Text style={styles.podiumScore}>{athletes[0].score}</Text>
            </View>

            {/* 3rd Place */}
            <View style={styles.podiumPosition}>
              <LinearGradient
                colors={getRankColors(3)}
                style={[styles.podiumCard, styles.thirdPlace]}
              >
                <Image source={{ uri: athletes[2].avatar }} style={styles.podiumAvatar} />
                <View style={styles.podiumRank}>
                  {getRankIcon(3)}
                </View>
              </LinearGradient>
              <Text style={styles.podiumName}>{athletes[2].name.split(' ')[0]}</Text>
              <Text style={styles.podiumScore}>{athletes[2].score}</Text>
            </View>
          </View>
        </Animated.View>

        {/* Full Rankings */}
        <Animated.View entering={FadeInDown.duration(800).delay(800)} style={styles.rankingsContainer}>
          <Text style={styles.rankingsTitle}>Full Rankings</Text>
          <View style={styles.rankingsList}>
            {athletes.map((athlete, index) => (
              <Animated.View 
                key={athlete.id}
                entering={FadeInUp.duration(600).delay(100 * index)}
                style={styles.rankingCard}
              >
                <View style={styles.rankingLeft}>
                  <View style={styles.rankNumber}>
                    <Text style={styles.rankNumberText}>{athlete.rank}</Text>
                  </View>
                  <Image source={{ uri: athlete.avatar }} style={styles.rankingAvatar} />
                  <View style={styles.athleteInfo}>
                    <Text style={styles.athleteName}>{athlete.name}</Text>
                    <Text style={styles.athleteDetails}>{athlete.region} • {athlete.sport}</Text>
                  </View>
                </View>
                
                <View style={styles.rankingRight}>
                  <Text style={styles.athleteScore}>{athlete.score}</Text>
                  <Text style={[
                    styles.athleteImprovement,
                    { color: athlete.improvement > 0 ? '#10B981' : '#EF4444' }
                  ]}>
                    {athlete.improvement > 0 ? '+' : ''}{athlete.improvement}%
                  </Text>
                </View>
              </Animated.View>
            ))}
          </View>
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
    paddingVertical: 20,
    backgroundColor: 'transparent',
  },
  headerTitle: {
    fontFamily: 'Poppins-Bold',
    fontSize: 28,
    color: '#FFFFFF',
  },
  headerSubtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#A5B4FC',
    marginTop: 8,
  },
  categoriesContainer: {
    paddingVertical: 20,
    backgroundColor: 'transparent',
  },
  categoriesScroll: {
    paddingLeft: 20,
  },
  categoryCard: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 12,
    backgroundColor: '#0B1220',
    marginRight: 12,
    gap: 8,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.25)'
  },
  categoryText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
  },
  filtersContainer: {
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 20,
    backgroundColor: '#FFFFFF',
  },
  filtersRow: {
    flexDirection: 'row',
    gap: 8,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#0B1220',
    borderWidth: 1,
    borderColor: 'rgba(99,102,241,0.18)'
  },
  activeFilter: {
    backgroundColor: '#7C3AED',
  },
  filterText: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: '#CBD5E1',
  },
  activeFilterText: {
    color: '#FFFFFF',
  },
  podiumContainer: {
    paddingHorizontal: 20,
    paddingVertical: 32,
  },
  podium: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    gap: 12,
  },
  podiumPosition: {
    alignItems: 'center',
    flex: 1,
  },
  podiumCard: {
    width: 80,
    borderRadius: 16,
    padding: 12,
    alignItems: 'center',
    position: 'relative',
    marginBottom: 12,
  },
  firstPlace: {
    height: 100,
  },
  secondPlace: {
    height: 85,
  },
  thirdPlace: {
    height: 70,
  },
  podiumAvatar: {
    width: 56,
    height: 56,
    borderRadius: 28,
    marginBottom: 8,
  },
  podiumRank: {
    position: 'absolute',
    top: -8,
    right: -8,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 4,
  },
  podiumName: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 14,
    color: '#111827',
    textAlign: 'center',
  },
  podiumScore: {
    fontFamily: 'Inter-Bold',
    fontSize: 12,
    color: '#6B7280',
    marginTop: 2,
  },
  rankingsContainer: {
    paddingHorizontal: 20,
  },
  rankingsTitle: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 20,
    color: '#FFFFFF',
    marginBottom: 16,
  },
  rankingsList: {
    gap: 12,
  },
  rankingCard: {
    backgroundColor: '#0B1220',
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 4,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.25)'
  },
  rankingLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  rankNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#1E1B4B',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  rankNumberText: {
    fontFamily: 'Poppins-Bold',
    fontSize: 14,
    color: '#FFFFFF',
  },
  rankingAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    marginRight: 12,
  },
  athleteInfo: {
    flex: 1,
  },
  athleteName: {
    fontFamily: 'Poppins-SemiBold',
    fontSize: 16,
    color: '#FFFFFF',
  },
  athleteDetails: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#94A3B8',
    marginTop: 2,
  },
  rankingRight: {
    alignItems: 'flex-end',
  },
  athleteScore: {
    fontFamily: 'Poppins-Bold',
    fontSize: 18,
    color: '#FFFFFF',
  },
  athleteImprovement: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 12,
    marginTop: 2,
  },
});
