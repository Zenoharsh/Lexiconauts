import React, { useEffect } from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { useSharedValue, withRepeat, withTiming, Easing, useAnimatedStyle } from 'react-native-reanimated';

export function AnimatedGlowIcon({
  size = 48,
  glowColors = ['#FF8A00', '#FFD700'],
  bgColors = ['#3B0764', '#0B1220'],
  children,
}: {
  size?: number;
  glowColors?: string[];
  bgColors?: string[];
  children: React.ReactNode;
}) {
  const pulse = useSharedValue(1);

  useEffect(() => {
    pulse.value = withRepeat(withTiming(1.08, { duration: 1400, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, [pulse]);

  const glowStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulse.value }],
    opacity: 0.9,
  }));

  return (
    <View style={{ width: size + 18, height: size + 18, alignItems: 'center', justifyContent: 'center' }}>
      <Animated.View style={[StyleSheet.absoluteFill, glowStyle]}> 
        <LinearGradient
          colors={glowColors}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={{ ...StyleSheet.absoluteFillObject, borderRadius: (size + 18) / 2, opacity: 0.38 }}
        />
      </Animated.View>
      <View style={{ width: size, height: size, borderRadius: size / 2, overflow: 'hidden' }}>
        <LinearGradient colors={bgColors} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          {typeof children === 'string' ? <Text style={{ color: '#FFFFFF', fontSize: size * 0.5 }}>{children}</Text> : children}
        </LinearGradient>
      </View>
    </View>
  );
}

export function FireProgress({ progress }: { progress: number }) {
  return (
    <View style={styles.progressBarContainer}>
      <View style={[styles.progressDynamic, { width: `${progress}%` }]}> 
        <LinearGradient colors={['#FF8A00', '#FFD700']} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={StyleSheet.absoluteFill} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
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
});
