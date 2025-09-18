import React, { useCallback, useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ActivityIndicator, Dimensions, KeyboardAvoidingView, Platform, StyleSheet, Text, TextInput, TouchableOpacity, View, Modal, ScrollView } from 'react-native';
import { Video, AVPlaybackStatus, ResizeMode, VideoReadyForDisplayEvent } from 'expo-av';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { Eye, EyeOff, LogIn, UserPlus } from 'lucide-react-native';
import SportifyLogo from './SportifyLogo';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CARD_WIDTH = Math.min(SCREEN_WIDTH - 32, 480);

type Mode = 'login' | 'signup';

export default function AuthScreen() {
  const router = useRouter();
  const videoRef = useRef<Video>(null);
  const [mode, setMode] = useState<Mode>('signup');
  const [videoReady, setVideoReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [showPassword2, setShowPassword2] = useState(false);

  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState<'Male' | 'Female' | ''>('');
  const [language, setLanguage] = useState<string>('');
  const [languageOpen, setLanguageOpen] = useState(false);

  const languages = useMemo(
    () => [
      'English',
      'Hindi',
      'Assamese',
      'Bengali',
      'Bodo',
      'Dogri',
      'Gujarati',
      'Kannada',
      'Kashmiri',
      'Konkani',
      'Maithili',
      'Malayalam',
      'Manipuri (Meitei)',
      'Marathi',
      'Nepali',
      'Odia (Oriya)',
      'Punjabi',
      'Sanskrit',
      'Santali',
      'Sindhi',
      'Tamil',
      'Telugu',
      'Urdu',
    ],
    []
  );

  const videoSource = useMemo(() => ({ uri: 'https://videos.pexels.com/video-files/6084029/6084029-hd_1920_1080_25fps.mp4' }), []);

  const onReady = useCallback((e: VideoReadyForDisplayEvent) => setVideoReady(true), []);

  const onPlaybackStatusUpdate = useCallback((status: AVPlaybackStatus) => {
    // Looping handled by player
  }, []);

  const switchMode = (next: Mode) => {
    Haptics.selectionAsync();
    setMode(next);
    setError(null);
  };

  const submitLogin = async () => {
    setError(null);
    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password.');
      return;
    }
    setLoading(true);
    try {
      await Haptics.selectionAsync();
      await new Promise((r) => setTimeout(r, 700));
      router.replace('/');
    } catch (e) {
      setError('Unable to sign in. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const submitSignup = async () => {
    setError(null);
    if (!fullName.trim()) return setError('Please enter your full name.');
    if (!email.trim()) return setError('Please enter your email.');
    if (!password.trim()) return setError('Please create a password.');
    if (password !== confirmPassword) return setError('Passwords do not match.');
    const ageNum = Number(age);
    if (!age || Number.isNaN(ageNum) || ageNum <= 0) return setError('Please enter a valid age.');
    if (!gender) return setError('Please select a gender.');
    if (!language) return setError('Please select a language.');

    setLoading(true);
    try {
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      await new Promise((r) => setTimeout(r, 900));
      router.replace('/');
    } catch (e) {
      setError('Unable to sign up. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const GenderOption = ({ value }: { value: 'Male' | 'Female' }) => (
    <TouchableOpacity
      onPress={() => setGender(value)}
      style={[styles.genderPill, gender === value && styles.genderPillActive]}
      accessibilityRole="button"
      accessibilityLabel={`Select ${value}`}
    >
      <Text
        style={[
          styles.genderText,
          value === 'Female' && styles.genderTextFemale,
          gender === value && styles.genderTextActive,
        ]}
        numberOfLines={1}
        ellipsizeMode="clip"
      >
        {value}
      </Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Video
          ref={videoRef}
          style={[StyleSheet.absoluteFill, styles.video]}
          source={videoSource}
          resizeMode={ResizeMode.COVER}
          shouldPlay
          isMuted
          isLooping
          rate={0.55}
          onReadyForDisplay={onReady}
          onPlaybackStatusUpdate={onPlaybackStatusUpdate}
        />
        <LinearGradient colors={["rgba(2,6,23,0.75)", "rgba(17,24,39,0.4)", "rgba(2,6,23,0.9)"]} style={StyleSheet.absoluteFill} />

        {!videoReady && (
          <View style={styles.videoLoader} pointerEvents="none">
            <ActivityIndicator color="#FFFFFF" />
          </View>
        )}

        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.formWrapper}>
          <BlurView intensity={40} tint="dark" style={styles.card}>
            <View style={styles.segmentWrap}>
              <TouchableOpacity onPress={() => switchMode('signup')} style={[styles.segment, mode === 'signup' && styles.segmentActive]} accessibilityRole="button" accessibilityLabel="Sign up">
                <UserPlus color={mode === 'signup' ? '#0B1220' : '#fff'} size={16} />
                <Text style={[styles.segmentText, mode === 'signup' && styles.segmentTextActive]}>Sign Up</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => switchMode('login')} style={[styles.segment, mode === 'login' && styles.segmentActive]} accessibilityRole="button" accessibilityLabel="Log in">
                <LogIn color={mode === 'login' ? '#0B1220' : '#fff'} size={16} />
                <Text style={[styles.segmentText, mode === 'login' && styles.segmentTextActive]}>Log In</Text>
              </TouchableOpacity>
            </View>

            {mode === 'login' ? (
              <>
                <Text style={styles.brand}>SPORTIFY</Text>
                <Text style={styles.subtitle}>Log in to track your performance and climb the leaderboard.</Text>
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
                      onSubmitEditing={submitLogin}
                    />
                    <TouchableOpacity onPress={() => setShowPassword((s) => !s)} style={styles.eyeBtn} accessibilityRole="button" accessibilityLabel={showPassword ? 'Hide password' : 'Show password'}>
                      {showPassword ? <EyeOff color="#fff" size={20} /> : <Eye color="#fff" size={20} />}
                    </TouchableOpacity>
                  </View>
                </View>
                {error ? <Text style={styles.errorText}>{error}</Text> : null}
                <TouchableOpacity style={[styles.button, loading && styles.buttonDisabled]} onPress={submitLogin} disabled={loading} accessibilityRole="button" accessibilityLabel="Log in">
                  {loading ? <ActivityIndicator color="#0B1220" /> : (
                    <View style={styles.buttonInner}>
                      <LogIn color="#0B1220" size={18} />
                      <Text style={styles.buttonText}>Log In</Text>
                    </View>
                  )}
                </TouchableOpacity>
                <TouchableOpacity onPress={() => router.push('/forgot-password')} accessibilityRole="button" accessibilityLabel="Forgot password">
                  <Text style={styles.linkText}>Forgot password?</Text>
                </TouchableOpacity>
              </>
            ) : (
              <>
                <Text style={styles.brand}>SPORTIFY</Text>
                <Text style={styles.subtitle}>Sign up to personalize your experience and compete with friends.</Text>
                <View style={styles.inputWrap}>
                  <Text style={styles.label}>Full Name</Text>
                  <TextInput
                    value={fullName}
                    onChangeText={setFullName}
                    placeholder="Your full name"
                    placeholderTextColor="rgba(255,255,255,0.5)"
                    autoCapitalize="words"
                    style={styles.input}
                    returnKeyType="next"
                  />
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
                <View style={styles.row}>
                  <View style={[styles.inputWrap, styles.half]}>
                    <Text style={styles.label}>Password</Text>
                    <View style={styles.passwordRow}>
                      <TextInput
                        value={password}
                        onChangeText={setPassword}
                        placeholder="Create password"
                        placeholderTextColor="rgba(255,255,255,0.5)"
                        secureTextEntry={!showPassword}
                        autoCapitalize="none"
                        textContentType="newPassword"
                        style={[styles.input, styles.passwordInput]}
                        returnKeyType="next"
                      />
                      <TouchableOpacity onPress={() => setShowPassword((s) => !s)} style={styles.eyeBtn} accessibilityRole="button" accessibilityLabel={showPassword ? 'Hide password' : 'Show password'}>
                        {showPassword ? <EyeOff color="#fff" size={20} /> : <Eye color="#fff" size={20} />}
                      </TouchableOpacity>
                    </View>
                  </View>
                  <View style={[styles.inputWrap, styles.half]}>
                    <Text style={styles.label}>Confirm</Text>
                    <View style={styles.passwordRow}>
                      <TextInput
                        value={confirmPassword}
                        onChangeText={setConfirmPassword}
                        placeholder="Confirm password"
                        placeholderTextColor="rgba(255,255,255,0.5)"
                        secureTextEntry={!showPassword2}
                        autoCapitalize="none"
                        textContentType="password"
                        style={[styles.input, styles.passwordInput]}
                        returnKeyType="next"
                      />
                      <TouchableOpacity onPress={() => setShowPassword2((s) => !s)} style={styles.eyeBtn} accessibilityRole="button" accessibilityLabel={showPassword2 ? 'Hide password' : 'Show password'}>
                        {showPassword2 ? <EyeOff color="#fff" size={20} /> : <Eye color="#fff" size={20} />}
                      </TouchableOpacity>
                    </View>
                  </View>
                </View>
                <View style={styles.row}>
                  <View style={[styles.inputWrap, styles.half]}>
                    <Text style={styles.label}>Age</Text>
                    <TextInput
                      value={age}
                      onChangeText={(t) => setAge(t.replace(/[^0-9]/g, ''))}
                      placeholder="18"
                      placeholderTextColor="rgba(255,255,255,0.5)"
                      keyboardType="number-pad"
                      style={styles.input}
                      returnKeyType="next"
                    />
                  </View>
                  <View style={[styles.inputWrap, styles.half]}>
                    <Text style={styles.label}>Gender</Text>
                    <View style={styles.genderRow}>
                      <GenderOption value="Male" />
                      <GenderOption value="Female" />
                    </View>
                  </View>
                </View>
                <View style={styles.inputWrap}>
                  <Text style={styles.label}>Preferred Language</Text>
                  <TouchableOpacity
                    onPress={() => setLanguageOpen(true)}
                    style={[styles.input, styles.selectInput]}
                    accessibilityRole="button"
                    accessibilityLabel="Select preferred language"
                  >
                    <Text style={language ? styles.selectText : styles.selectPlaceholder} numberOfLines={1}>
                      {language || 'Select language'}
                    </Text>
                  </TouchableOpacity>
                </View>

                <Modal
                  transparent
                  visible={languageOpen}
                  animationType="fade"
                  onRequestClose={() => setLanguageOpen(false)}
                >
                  <View style={styles.modalOverlay}>
                    <View style={styles.modalCard}>
                      <Text style={styles.modalTitle}>Select Language</Text>
                      <ScrollView style={styles.modalList}>
                        {languages.map((lang) => (
                          <TouchableOpacity
                            key={lang}
                            onPress={() => { setLanguage(lang); setLanguageOpen(false); }}
                            style={styles.modalItem}
                            accessibilityRole="button"
                            accessibilityLabel={`Choose ${lang}`}
                          >
                            <Text style={styles.modalItemText}>{lang}</Text>
                          </TouchableOpacity>
                        ))}
                      </ScrollView>
                      <TouchableOpacity onPress={() => setLanguageOpen(false)} style={styles.modalCancel} accessibilityRole="button" accessibilityLabel="Cancel">
                        <Text style={styles.modalCancelText}>Cancel</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                </Modal>

                {error ? <Text style={styles.errorText}>{error}</Text> : null}
                <TouchableOpacity style={[styles.button, loading && styles.buttonDisabled]} onPress={submitSignup} disabled={loading} accessibilityRole="button" accessibilityLabel="Create account">
                  {loading ? <ActivityIndicator color="#0B1220" /> : (
                    <View style={styles.buttonInner}>
                      <UserPlus color="#0B1220" size={18} />
                      <Text style={styles.buttonText}>Create Account</Text>
                    </View>
                  )}
                </TouchableOpacity>
              </>
            )}
          </BlurView>
          <View style={styles.footer} pointerEvents="none">
            <Text style={styles.footerText}>Cinematic arena lights set the stage — perform at your best.</Text>
          </View>
        </KeyboardAvoidingView>

      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#000' },
  container: { flex: 1, justifyContent: 'center' },
  videoLoader: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
  formWrapper: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16 },
  card: { width: CARD_WIDTH, borderRadius: 20, overflow: 'hidden', padding: 20, borderWidth: 1, borderColor: 'rgba(255,255,255,0.12)', backgroundColor: 'rgba(15, 23, 42, 0.35)' },
  segmentWrap: { flexDirection: 'row', backgroundColor: 'rgba(255,255,255,0.08)', padding: 4, borderRadius: 12, marginBottom: 12 },
  segment: { flex: 1, height: 40, borderRadius: 10, alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 8 },
  segmentActive: { backgroundColor: '#F59E0B' },
  segmentText: { fontFamily: 'Inter-Medium', color: '#fff', fontSize: 14 },
  segmentTextActive: { color: '#0B1220', fontFamily: 'Inter-SemiBold' },
  brand: { fontFamily: 'Poppins-Bold', fontSize: 26, color: '#FFFFFF', letterSpacing: 0.5, marginTop: 4 },
  subtitle: { marginTop: 6, fontFamily: 'Inter-Regular', fontSize: 14, color: 'rgba(255,255,255,0.85)' },
  inputWrap: { width: '100%', marginTop: 12 },
  label: { fontFamily: 'Inter-Medium', fontSize: 12, color: 'rgba(255,255,255,0.85)', marginBottom: 6 },
  input: { width: '100%', height: 48, borderRadius: 12, paddingHorizontal: 14, backgroundColor: 'rgba(255,255,255,0.12)', color: '#FFF', borderWidth: 1, borderColor: 'rgba(255,255,255,0.18)', fontFamily: 'Inter-Regular', fontSize: 16 },
  selectInput: { justifyContent: 'center' },
  selectText: { color: '#FFF', fontFamily: 'Inter-Regular', fontSize: 16 },
  selectPlaceholder: { color: 'rgba(255,255,255,0.5)', fontFamily: 'Inter-Regular', fontSize: 16 },
  passwordRow: { position: 'relative', width: '100%', justifyContent: 'center' },
  passwordInput: { paddingRight: 44 },
  eyeBtn: { position: 'absolute', right: 10, height: 48, width: 36, alignItems: 'center', justifyContent: 'center' },
  row: { flexDirection: 'row', gap: 12 },
  half: { flex: 1 },
  genderRow: { flexDirection: 'row', gap: 8, alignItems: 'center', justifyContent: 'space-between' },
  genderPill: { paddingHorizontal: 14, height: 40, borderRadius: 10, borderWidth: 1, borderColor: 'rgba(255,255,255,0.18)', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.12)' },
  genderPillActive: { backgroundColor: '#F59E0B', borderColor: '#F59E0B' },
  genderText: { color: '#fff', fontFamily: 'Inter-Medium', fontSize: 14 },
  genderTextActive: { color: '#0B1220', fontFamily: 'Inter-SemiBold' },
  errorText: { marginTop: 10, color: '#FCA5A5', fontFamily: 'Inter-Medium' },
  button: { marginTop: 16, width: '100%', height: 50, borderRadius: 14, alignItems: 'center', justifyContent: 'center', backgroundColor: '#F59E0B' },
  buttonDisabled: { opacity: 0.7 },
  buttonInner: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  buttonText: { fontFamily: 'Inter-SemiBold', color: '#0B1220', fontSize: 16 },
  linkText: { marginTop: 12, color: 'rgba(255,255,255,0.95)', fontFamily: 'Inter-Medium' },
  footer: { position: 'absolute', bottom: 24, width: '100%', alignItems: 'center', paddingHorizontal: 16 },
  footerText: { color: 'rgba(255,255,255,0.75)', fontFamily: 'Inter-Regular', fontSize: 12, textAlign: 'center' },
  logoBg: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center', opacity: 0.4 },
  logoImage: { width: 280, height: 280 },
  video: { transform: [{ scale: 0.96 }] },
  genderTextFemale: { fontSize: 13 },
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', alignItems: 'center', justifyContent: 'center', padding: 16 },
  modalCard: { width: Math.min(SCREEN_WIDTH - 32, 480), maxHeight: '70%', borderRadius: 16, padding: 16, backgroundColor: 'rgba(15,23,42,0.98)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.12)' },
  modalTitle: { fontFamily: 'Inter-SemiBold', color: '#fff', fontSize: 16, marginBottom: 8 },
  modalList: { marginVertical: 8 },
  modalItem: { paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.08)' },
  modalItemText: { color: '#fff', fontFamily: 'Inter-Regular', fontSize: 16 },
  modalCancel: { marginTop: 8, alignSelf: 'flex-end', paddingVertical: 8, paddingHorizontal: 12, borderRadius: 8, backgroundColor: 'rgba(255,255,255,0.12)' },
  modalCancelText: { color: '#fff', fontFamily: 'Inter-Medium' },
});
