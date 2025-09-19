import React, { useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { KeyboardAvoidingView, Platform, StyleSheet, Text, TextInput, TouchableOpacity, View, ActivityIndicator } from 'react-native';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';

export default function ForgotPassword() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  const submit = async () => {
    if (!email.trim()) return;
    setLoading(true);
    try {
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      await new Promise((r) => setTimeout(r, 800));
      setSent(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={StyleSheet.absoluteFill}>
        <LinearGradient colors={["#0B1220", "#1E1B4B"]} style={StyleSheet.absoluteFill} />
      </View>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.wrap}>
        <BlurView intensity={40} tint="dark" style={styles.card}>
          <Text style={styles.title}>Forgot Password</Text>
          {sent ? (
            <>
              <Text style={styles.subtitle}>If an account exists for {email}, a reset link has been sent.</Text>
              <TouchableOpacity onPress={() => router.back()} style={styles.button} accessibilityRole="button" accessibilityLabel="Back to login">
                <Text style={styles.buttonText}>Back to Login</Text>
              </TouchableOpacity>
            </>
          ) : (
            <>
              <Text style={styles.subtitle}>Enter your email to receive a password reset link.</Text>
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
                  returnKeyType="go"
                  onSubmitEditing={submit}
                />
              </View>
              <TouchableOpacity onPress={submit} style={[styles.button, loading && styles.buttonDisabled]} disabled={loading} accessibilityRole="button" accessibilityLabel="Send reset link">
                {loading ? <ActivityIndicator color="#0B1220" /> : <Text style={styles.buttonText}>Send Reset Link</Text>}
              </TouchableOpacity>
            </>
          )}
        </BlurView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#0B1220' },
  wrap: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 16 },
  card: { width: 420, maxWidth: '100%', borderRadius: 20, overflow: 'hidden', padding: 20, borderWidth: 1, borderColor: 'rgba(255,255,255,0.12)', backgroundColor: 'rgba(15, 23, 42, 0.35)' },
  title: { fontFamily: 'Poppins-Bold', fontSize: 24, color: '#fff' },
  subtitle: { marginTop: 8, fontFamily: 'Inter-Regular', color: 'rgba(255,255,255,0.9)' },
  inputWrap: { width: '100%', marginTop: 16 },
  label: { fontFamily: 'Inter-Medium', fontSize: 12, color: 'rgba(255,255,255,0.85)', marginBottom: 6 },
  input: { width: '100%', height: 48, borderRadius: 12, paddingHorizontal: 14, backgroundColor: 'rgba(255,255,255,0.12)', color: '#FFF', borderWidth: 1, borderColor: 'rgba(255,255,255,0.18)', fontFamily: 'Inter-Regular', fontSize: 16 },
  button: { marginTop: 18, width: '100%', height: 50, borderRadius: 14, alignItems: 'center', justifyContent: 'center', backgroundColor: '#F59E0B' },
  buttonDisabled: { opacity: 0.7 },
  buttonText: { fontFamily: 'Inter-SemiBold', color: '#0B1220', fontSize: 16 },
});
