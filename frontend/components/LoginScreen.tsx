import React, { useRef, useState, useCallback } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StyleSheet, View, Text, TextInput, TouchableOpacity, KeyboardAvoidingView, Platform, Dimensions, ActivityIndicator } from 'react-native';
import { Video, ResizeMode, AVPlaybackStatus, VideoReadyForDisplayEvent } from 'expo-av';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { Eye, EyeOff, LogIn } from 'lucide-react-native';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function LoginScreen() {
  const router = useRouter();
  const videoRef = useRef<Video>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onReady = useCallback((e: VideoReadyForDisplayEvent) => {
    setVideoReady(true);
  }, []);

  const onPlaybackStatusUpdate = useCallback((status: AVPlaybackStatus) => {
    // Intentionally no-op; hook kept for potential analytics or UI reactions
  }, []);

  const handleLogin = async () => {
    setError(null);
    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password.');
      return;
    }
    setLoading(true);
    try {
      await Haptics.selectionAsync();
      // Simulate authentication delay; replace with real auth when available
      await new Promise((r) => setTimeout(r, 600));
      router.replace('/');
    } catch (e) {
      setError('Unable to sign in. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const togglePassword = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setShowPassword((s) => !s);
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Video
          ref={videoRef}
          style={StyleSheet.absoluteFill}
          source={{ uri: 'https://videos.pexels.com/video-files/6084022/6084022-hd_1920_1080_25fps.mp4' }}
          resizeMode={ResizeMode.COVER}
          shouldPlay
          isLooping
          isMuted
          rate={0.6}
          onReadyForDisplay={onReady}
          onPlaybackStatusUpdate={onPlaybackStatusUpdate}
        />
        <LinearGradient
          colors={["rgba(0,0,0,0.65)", "rgba(0,0,0,0.35)", "rgba(0,0,0,0.75)"]}
          style={StyleSheet.absoluteFill}
        />

        {!videoReady && (
          <View style={styles.videoLoader} pointerEvents="none">
            <ActivityIndicator color="#FFFFFF" />
          </View>
        )}

        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={styles.formWrapper}
          contentContainerStyle={styles.formContent}
        >
          <BlurView intensity={40} tint="dark" style={styles.card}>
            <View style={styles.headerWrap}>
              <Text style={styles.brand}>GoalSight</Text>
              <Text style={styles.subtitle}>Sign in to track your shot analytics and compete on the leaderboard.</Text>
            </View>

            <View style={styles.inputWrap}>
              <Text style={styles.label}>Email</Text>
              <TextInput
                value={email}
                onChangeText={setEmail}
                placeholder="you@example.com"
                placeholderTextColor="rgba(255,255,255,0.5)"
                keyboardType="email-address"
                autoCapitalize="none"
                autoComplete="email"
                textContentType="emailAddress"
                style={styles.input}
                returnKeyType="next"
              />
            </View>

            <View style={styles.inputWrap}>
              <Text style={styles.label}>Password</Text>
              <View style={styles.passwordRow}>
                <TextInput
                  value={password}
                  onChangeText={setPassword}
                  placeholder="Your secure password"
                  placeholderTextColor="rgba(255,255,255,0.5)"
                  secureTextEntry={!showPassword}
                  autoCapitalize="none"
                  textContentType="password"
                  style={[styles.input, styles.passwordInput]}
                  returnKeyType="go"
                  onSubmitEditing={handleLogin}
                />
                <TouchableOpacity accessibilityRole="button" accessibilityLabel={showPassword ? 'Hide password' : 'Show password'} onPress={togglePassword} style={styles.eyeBtn}>
                  {showPassword ? <EyeOff color="#fff" size={20} /> : <Eye color="#fff" size={20} />}
                </TouchableOpacity>
              </View>
            </View>

            {error ? (
              <Text style={styles.errorText}>{error}</Text>
            ) : null}

            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={handleLogin}
              disabled={loading}
              accessibilityRole="button"
              accessibilityLabel="Sign in"
            >
              {loading ? (
                <ActivityIndicator color="#0B1220" />
              ) : (
                <View style={styles.buttonInner}>
                  <LogIn color="#0B1220" size={18} />
                  <Text style={styles.buttonText}>Sign In</Text>
                </View>
              )}
            </TouchableOpacity>

            <TouchableOpacity
              onPress={() => router.replace('/')}
              accessibilityRole="button"
              accessibilityLabel="Skip sign in"
            >
              <Text style={styles.skipText}>Skip for now</Text>
            </TouchableOpacity>
          </BlurView>

          <View style={styles.footer} pointerEvents="none">
            <Text style={styles.footerText}>Shot captured from goal net — slow motion for dramatic effect.</Text>
          </View>
        </KeyboardAvoidingView>
      </View>
    </SafeAreaView>
  );
}

const CARD_WIDTH = Math.min(SCREEN_WIDTH - 32, 440);

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#000',
  },
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  videoLoader: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  formWrapper: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
  },
  formContent: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    width: CARD_WIDTH,
    borderRadius: 20,
    overflow: 'hidden',
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
    backgroundColor: 'rgba(15, 23, 42, 0.35)',
  },
  headerWrap: {
    marginBottom: 16,
  },
  brand: {
    fontFamily: 'Poppins-Bold',
    fontSize: 28,
    color: '#FFFFFF',
    letterSpacing: 0.5,
  },
  subtitle: {
    marginTop: 6,
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
  },
  inputWrap: {
    width: '100%',
    marginTop: 12,
  },
  label: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    marginBottom: 6,
  },
  input: {
    width: '100%',
    height: 48,
    borderRadius: 12,
    paddingHorizontal: 14,
    backgroundColor: 'rgba(255,255,255,0.12)',
    color: '#FFF',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.18)',
    fontFamily: 'Inter-Regular',
    fontSize: 16,
  },
  passwordRow: {
    position: 'relative',
    width: '100%',
    justifyContent: 'center',
  },
  passwordInput: {
    paddingRight: 44,
  },
  eyeBtn: {
    position: 'absolute',
    right: 10,
    height: 48,
    width: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorText: {
    marginTop: 10,
    color: '#FCA5A5',
    fontFamily: 'Inter-Medium',
  },
  button: {
    marginTop: 16,
    width: '100%',
    height: 50,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F59E0B',
  },
  buttonDisabled: {
    opacity: 0.7,
  },
  buttonInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  buttonText: {
    fontFamily: 'Inter-SemiBold',
    color: '#0B1220',
    fontSize: 16,
  },
  skipText: {
    marginTop: 12,
    color: 'rgba(255,255,255,0.9)',
    fontFamily: 'Inter-Medium',
  },
  footer: {
    position: 'absolute',
    bottom: 24,
    width: '100%',
    alignItems: 'center',
    paddingHorizontal: 16,
  },
  footerText: {
    color: 'rgba(255,255,255,0.7)',
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    textAlign: 'center',
  },
});
