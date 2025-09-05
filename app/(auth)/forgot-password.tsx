import React, { useState } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  TextInput, 
  Alert
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { Mail, ArrowLeft, Send } from 'lucide-react-native';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { router } from 'expo-router';

export default function ForgotPasswordScreen() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleResetPassword = async () => {
    if (!email) {
      Alert.alert('Error', 'Please enter your email address');
      return;
    }

    setIsLoading(true);
    
    // Simulate password reset process
    setTimeout(() => {
      setIsLoading(false);
      Alert.alert(
        'Reset Link Sent', 
        'Check your email for password reset instructions',
        [{ text: 'OK', onPress: () => router.back() }]
      );
    }, 2000);
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Background Gradient */}
      <LinearGradient
        colors={['#1E40AF', '#3B82F6', '#60A5FA']}
        style={styles.background}
      />

      <Animated.View entering={FadeInDown.duration(800)} style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <ArrowLeft size={24} color="#FFFFFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Reset Password</Text>
        <Text style={styles.headerSubtitle}>
          Enter your email address and we'll send you a link to reset your password
        </Text>
      </Animated.View>

      <Animated.View 
        entering={FadeInUp.duration(800).delay(200)}
        style={styles.formContainer}
      >
        <BlurView intensity={20} style={styles.formBlur}>
          <View style={styles.form}>
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

            <TouchableOpacity 
              style={styles.resetButton}
              onPress={handleResetPassword}
              disabled={isLoading}
            >
              <LinearGradient
                colors={['#10B981', '#059669']}
                style={styles.resetGradient}
              >
                {isLoading ? (
                  <Text style={styles.resetButtonText}>Sending...</Text>
                ) : (
                  <>
                    <Send size={20} color="#FFFFFF" />
                    <Text style={styles.resetButtonText}>Send Reset Link</Text>
                  </>
                )}
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </BlurView>
      </Animated.View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  background: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  header: {
    marginBottom: 40,
  },
  backButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  headerTitle: {
    fontFamily: 'Poppins-Bold',
    fontSize: 32,
    color: '#FFFFFF',
    marginBottom: 12,
  },
  headerSubtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#D1D5DB',
    lineHeight: 24,
  },
  formContainer: {
    borderRadius: 24,
    overflow: 'hidden',
  },
  formBlur: {
    padding: 24,
  },
  form: {
    width: '100%',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 16,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
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
  resetButton: {
    width: '100%',
    borderRadius: 16,
    overflow: 'hidden',
  },
  resetGradient: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 16,
    gap: 8,
  },
  resetButtonText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 16,
    color: '#FFFFFF',
  },
});