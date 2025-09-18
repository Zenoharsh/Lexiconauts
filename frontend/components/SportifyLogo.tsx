import React, { useEffect } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Stop, Path, Text as SvgText, G, Circle } from 'react-native-svg';
import Animated, { useSharedValue, withTiming, withRepeat, Easing, useAnimatedStyle, withDelay } from 'react-native-reanimated';

export default function SportifyLogo({ size = 160 }: { size?: number }) {
  const pulse = useSharedValue(0.92);
  const rotate = useSharedValue(0);

  useEffect(() => {
    pulse.value = withRepeat(withTiming(1.08, { duration: 1400, easing: Easing.inOut(Easing.quad) }), -1, true);
    rotate.value = withRepeat(withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.cubic) }), -1, true);
  }, [pulse, rotate]);

  const pulseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulse.value }],
    opacity: 1,
  }));

  const streakStyle = useAnimatedStyle(() => ({
    opacity: withDelay(300, withTiming(1, { duration: 400 })),
    transform: [
      { translateX: withTiming(0) },
      { rotate: `${(rotate.value - 0.5) * 6}deg` },
    ],
  }));

  const s = size;

  return (
    <View style={[styles.wrap, { width: s, height: s }]}> 
      <Animated.View style={[StyleSheet.absoluteFill, pulseStyle]}> 
        <Svg width={s} height={s} viewBox="0 0 200 200">
          <Defs>
            <SvgLinearGradient id="gradCool" x1="0" y1="0" x2="1" y2="1">
              <Stop offset="0%" stopColor="#60A5FA" />
              <Stop offset="55%" stopColor="#8B5CF6" />
              <Stop offset="100%" stopColor="#22D3EE" />
            </SvgLinearGradient>
            <SvgLinearGradient id="shine" x1="0" y1="0" x2="1" y2="0">
              <Stop offset="0%" stopColor="#ffffff" stopOpacity="1" />
              <Stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
            </SvgLinearGradient>
            <SvgLinearGradient id="textGrad" x1="0" y1="0" x2="1" y2="0">
              <Stop offset="0%" stopColor="#E5E7EB" />
              <Stop offset="100%" stopColor="#A5B4FC" />
            </SvgLinearGradient>
          </Defs>

          {/* Outer glow ring */}
          <Circle cx="100" cy="100" r="86" stroke="url(#gradCool)" strokeOpacity="0.45" strokeWidth="10" fill="none" />

          {/* Shield base */}
          <Path d="M100 10 L165 35 C168 90 150 140 100 180 C50 140 32 90 35 35 Z" fill="url(#gradCool)" />

          {/* Neon outline */}
          <Path d="M100 10 L165 35 C168 90 150 140 100 180 C50 140 32 90 35 35 Z" fill="none" stroke="#FFFFFF" strokeOpacity="0.25" strokeWidth="2" />

          {/* S mark with glow */}
          <Path d="M130 70 C130 55 115 48 98 48 C84 48 70 54 70 67 C70 86 92 88 102 92 C115 96 128 100 128 116 C128 130 113 140 96 140 C78 140 66 130 64 118" stroke="url(#gradCool)" strokeWidth="12" strokeLinecap="round" fill="none" />
          <Path d="M130 70 C130 55 115 48 98 48 C84 48 70 54 70 67 C70 86 92 88 102 92 C115 96 128 100 128 116 C128 130 113 140 96 140 C78 140 66 130 64 118" stroke="#0B1220" strokeWidth="6" strokeLinecap="round" fill="none" />

          {/* Title with gradient and stroke */}
          <SvgText x="100" y="195" textAnchor="middle" fontSize="28" fontWeight="800" fill="url(#textGrad)" stroke="#111827" strokeWidth="0.6" fontFamily="Poppins-Bold" letterSpacing="2">SPORTIFY</SvgText>

          {/* Shine */}
          <Path d="M35 35 L165 35" stroke="url(#shine)" strokeWidth="6" />

          {/* Glints */}
          <G opacity="0.85">
            <Path d="M60 58 L64 62 M64 58 L60 62" stroke="#FFFFFF" strokeWidth="1.5" strokeLinecap="round" />
            <Path d="M138 48 L142 52 M142 48 L138 52" stroke="#FFFFFF" strokeWidth="1.2" strokeLinecap="round" />
          </G>
        </Svg>
      </Animated.View>

      {/* Motion streak */}
      <Animated.View style={[StyleSheet.absoluteFill, streakStyle]}> 
        <Svg width={s} height={s} viewBox="0 0 200 200">
          <Path d="M-20 120 C40 90, 160 150, 220 120" stroke="#93C5FD" strokeOpacity="0.35" strokeWidth="8" fill="none" />
        </Svg>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    justifyContent: 'center',
  },
});
