import React, { useState, useRef, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  TextInput, 
  Dimensions, 
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Video, ResizeMode } from 'expo-av';
import { BlurView } from 'expo-blur';
import { 
  Mail, 
  Lock, 
  Eye, 
  EyeOff, 
  Play, 
  Trophy, 
  Target,
  ArrowRight,
  User,
  Smartphone
} from 'lucide-react-native';
import Animated, { 
  FadeInDown, 
  FadeInUp, 
  useSharedValue, 
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withTiming
} from 'react-native-reanimated';
import { router } from 'expo-router';

const { width, height } = Dimensions.get('window');

interface SportVideo {
  id: string;
  title: string;
  sport: string;
  uri: string;
  thumbnail: string;
}

export default function LoginScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const videoRef = useRef<Video>(null);

  // Animation values
  const logoScale = useSharedValue(1);
  const floatingY = useSharedValue(0);

  // Sports videos (using placeholder URLs - in production, use actual sports videos)
  const sportVideos: SportVideo[] = [
    {
      id: '1',
      title: 'Football Training',
      sport: 'Football',
      uri: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
      thumbnail: 'https://images.pexels.com/photos/274422/pexels-photo-274422.jpeg'
    },
    {
      id: '2',
      title: 'Basketball Skills',
      sport: 'Basketball',
      uri: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
      thumbnail: 'https://images.pexels.com/photos/358042/pexels-photo-358042.jpeg'
    },
    {
      id: '3',
      title: 'Athletic Performance',
      sport: 'Track & Field',
      uri: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
      thumbnail: 'https://images.pexels.com/photos/936094/pexels-photo-936094.jpeg'
    }
  ];

  useEffect(() => {
    // Logo pulsing animation
    logoScale.value = withRepeat(
      withTiming(1.1, { duration: 2000 }),
      -1,
      true
    );

    // Floating animation
    floatingY.value = withRepeat(
      withTiming(-10, { duration: 3000 }),
      -1,
      true
    );

    // Auto-switch videos every 10 seconds
    const interval = setInterval(() => {
      setCurrentVideoIndex((prev) => (prev + 1) % sportVideos.length);
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const logoAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: logoScale.value }],
  }));

  const floatingAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: floatingY.value }],
  }));

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    setIsLoading(true);
    
    // Simulate login process
    setTimeout(() => {
      setIsLoading(false);
      // Navigate to main app
      router.replace('/(tabs)');
    }, 2000);
  };

  const handleSocialLogin = (provider: string) => {
    Alert.alert('Social Login', `${provider} login will be implemented`);
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        {/* Background Video */}
        <View style={styles.videoContainer}>
          <Video
            ref={videoRef}
            source={{ uri: sportVideos[currentVideoIndex].uri }}
            style={styles.backgroundVideo}
            resizeMode={ResizeMode.COVER}
            shouldPlay
            isLooping
            isMuted
          />
          <LinearGradient
            colors={['rgba(0,0,0,0.7)', 'rgba(0,0,0,0.3)', 'rgba(0,0,0,0.8)']}
            style={styles.videoOverlay}
          />
        </View>

        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header Section */}
          <Animated.View 
            entering={FadeInDown.duration(1000)}
            style={[styles.header, floatingAnimatedStyle]}
          >
            <Animated.View style={[styles.logoContainer, logoAnimatedStyle]}>
              <LinearGradient
                colors={['#F59E0B', '#F97316']}
                style={styles.logoGradient}
              >
                <Trophy size={32} color="#FFFFFF" />
              </LinearGradient>
            </Animated.View>
            <Text style={styles.appTitle}>SportsTech AI</Text>
            <Text style={styles.appSubtitle}>Elite Athletic Assessment Platform</Text>
          </Animated.View>

          {/* Video Info Card */}
          <Animated.View 
            entering={FadeInUp.duration(800).delay(200)}
            style={styles.videoInfoCard}
          >
            <BlurView intensity={20} style={styles.blurContainer}>
              <View style={styles.videoInfo}>
                <Play size={16} color="#F59E0B" />
                <Text style={styles.videoTitle}>{sportVideos[currentVideoIndex].title}</Text>
              </View>
              <Text style={styles.videoSport}>{sportVideos[currentVideoIndex].sport}</Text>
            </BlurView>
          </Animated.View>

          {/* Login Form */}
          <Animated.View 
            entering={FadeInUp.duration(800).delay(400)}
            style={styles.formContainer}
          >
            <BlurView intensity={40} style={styles.formBlur}>
              <View style={styles.form}>
                <Text style={styles.formTitle}>Welcome Back</Text>
                <Text style={styles.formSubtitle}>Sign in to continue your athletic journey</Text>

                {/* Email Input */}
                <View style={styles.inputContainer}>
                  <View style={styles.inputIcon}>
                    <Mail size={20} color="#6B7280" />
                  </View>
                  <TextInput
                    style={styles.input}
                    placeholder="Email address"
                    placeholderTextColor="#9CA3AF"
                    value={email}
                    onChangeText={setEmail}
                    keyboardType="email-address"
                    autoCapitalize="none"
                  />
                </View>

                {/* Password Input */}
                <View style={styles.inputContainer}>
                  <View style={styles.inputIcon}>
                    <Lock size={20} color="#6B7280" />
                  </View>
                  <TextInput
                    style={styles.input}
                    placeholder="Password"
                    placeholderTextColor="#9CA3AF"
                    value={password}
                    onChangeText={setPassword}
                    secureTextEntry={!showPassword}
                  />
                  <TouchableOpacity 
                    style={styles.eyeIcon}
                    onPress={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff size={20} color="#6B7280" />
                    ) : (
                      <Eye size={20} color="#6B7280" />
                    )}
                  </TouchableOpacity>
                </View>

                {/* Forgot Password */}
                <TouchableOpacity 
                  style={styles.forgotPassword}
                  onPress={() => router.push('/(auth)/forgot-password')}
                >
                  <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
                </TouchableOpacity>

                {/* Login Button */}
                <TouchableOpacity 
                  style={styles.loginButton}
                  onPress={handleLogin}
                  disabled={isLoading}
                >
                  <LinearGradient
                    colors={['#1E40AF', '#3B82F6']}
                    style={styles.loginGradient}
                  >
                    {isLoading ? (
                      <Text style={styles.loginButtonText}>Signing In...</Text>
                    ) : (
                      <>
                        <Text style={styles.loginButtonText}>Sign In</Text>
                        <ArrowRight size={20} color="#FFFFFF" />
                      </>
                    )}
                  </LinearGradient>
                </TouchableOpacity>

                {/* Divider */}
                <View style={styles.divider}>
                  <View style={styles.dividerLine} />
                  <Text style={styles.dividerText}>or continue with</Text>
                  <View style={styles.dividerLine} />
                </View>

                {/* Social Login */}
                <View style={styles.socialContainer}>
                  <TouchableOpacity 
                    style={styles.socialButton}
                    onPress={() => handleSocialLogin('Google')}
                  >
                    <Text style={styles.socialButtonText}>G</Text>
                  </TouchableOpacity>
                  <TouchableOpacity 
                    style={styles.socialButton}
                    onPress={() => handleSocialLogin('Apple')}
                  >
                    <Smartphone size={20} color="#374151" />
                  </TouchableOpacity>
                  <TouchableOpacity 
                    style={styles.socialButton}
                    onPress={() => handleSocialLogin('Facebook')}
                  >
                    <Text style={styles.socialButtonText}>f</Text>
                  </TouchableOpacity>
                </View>

                {/* Sign Up Link */}
                <View style={styles.signupContainer}>
                  <Text style={styles.signupText}>Don't have an account? </Text>
                  <TouchableOpacity onPress={() => router.push('/(auth)/register')}>
                    <Text style={styles.signupLink}>Sign Up</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </BlurView>
          </Animated.View>

          {/* Features Preview */}
          <Animated.View 
            entering={FadeInUp.duration(800).delay(600)}
            style={styles.featuresContainer}
          >
            <View style={styles.featureItem}>
              <View style={styles.featureIcon}>
                <Target size={16} color="#10B981" />
              </View>
              <Text style={styles.featureText}>AI-Powered Analysis</Text>
            </View>
            <View style={styles.featureItem}>
              <View style={styles.featureIcon}>
                <Trophy size={16} color="#F59E0B" />
              </View>
              <Text style={styles.featureText}>Performance Tracking</Text>
            </View>
            <View style={styles.featureItem}>
              <View style={styles.featureIcon}>
                <User size={16} color="#8B5CF6" />
              </View>
              <Text style={styles.featureText}>Personalized Training</Text>
            </View>
          </Animated.View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  keyboardView: {
    flex: 1,
  },
  videoContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  backgroundVideo: {
    width: width,
    height: height,
  },
  videoOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  logoContainer: {
    marginBottom: 20,
  },
  logoGradient: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#F59E0B',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    elevation: 8,
  },
  appTitle: {
    fontFamily: 'Poppins-Bold',
    fontSize: 32,
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 4,
  },
  appSubtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#E5E7EB',
    textAlign: 'center',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  videoInfoCard: {
    marginBottom: 30,
    borderRadius: 16,
    overflow: 'hidden',
  },
  blurContainer: {
    padding: 16,
  },
  videoInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  videoTitle: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
    color: '#FFFFFF',
    marginLeft: 8,
  },
  videoSport: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#D1D5DB',
  },
  formContainer: {
    borderRadius: 24,
    overflow: 'hidden',
    marginBottom: 30,
  },
  formBlur: {
    padding: 24,
  },
  form: {
    alignItems: 'center',
  },
  formTitle: {
    fontFamily: 'Poppins-Bold',
    fontSize: 28,
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 8,
  },
  formSubtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#D1D5DB',
    textAlign: 'center',
    marginBottom: 32,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
    width: '100%',
  },
  inputIcon: {
    padding: 16,
  },
  input: {
    flex: 1,
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#FFFFFF',
    paddingVertical: 16,
    paddingRight: 16,
  },
  eyeIcon: {
    padding: 16,
  },
  forgotPassword: {
    alignSelf: 'flex-end',
    marginBottom: 24,
  },
  forgotPasswordText: {
    fontFamily: 'Inter-Medium',
    fontSize: 14,
    color: '#3B82F6',
  },
  loginButton: {
    width: '100%',
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 24,
  },
  loginGradient: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 16,
    gap: 8,
  },
  loginButtonText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 16,
    color: '#FFFFFF',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
    width: '100%',
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  dividerText: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#9CA3AF',
    marginHorizontal: 16,
  },
  socialContainer: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 24,
  },
  socialButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(255,255,255,0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  socialButtonText: {
    fontFamily: 'Poppins-Bold',
    fontSize: 20,
    color: '#374151',
  },
  signupContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  signupText: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#D1D5DB',
  },
  signupLink: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
    color: '#3B82F6',
  },
  featuresContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
  },
  featureItem: {
    alignItems: 'center',
    flex: 1,
  },
  featureIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  featureText: {
    fontFamily: 'Inter-Medium',
    fontSize: 12,
    color: '#E5E7EB',
    textAlign: 'center',
  },
});