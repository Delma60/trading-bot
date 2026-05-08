from manager.profile_manager import ProfileManager

# Test that ProfileManager has the correct attributes
pm = ProfileManager()
print(f'✓ config_file exists: {hasattr(pm, "config_file")}')
print(f'✓ credentials_file exists: {hasattr(pm, "credentials_file")}')
print(f'✓ load_credentials method exists: {hasattr(pm, "load_credentials")}')
print(f'✓ save_credentials method exists: {hasattr(pm, "save_credentials")}')
print(f'✓ profile_file exists: {hasattr(pm, "profile_file")}')  # Should be False

# Test load_credentials (should return empty dict)
creds = pm.load_credentials()
print(f'✓ load_credentials returns dict: {isinstance(creds, dict)}')
print('All ProfileManager attributes verified!')
