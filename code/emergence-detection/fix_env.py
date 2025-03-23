#!/usr/bin/env python3
"""
Script to fix the .env file configuration for HAWKEYEZ application.
This ensures the correct email address is being used.
"""

import os
import sys

def fix_env_file():
    # Define the correct configuration
    correct_email = "mangalarapumanu@gmail.com"
    app_password = "jjjg kxuq eyhx agre"  # Use the current app password
    emergency_contacts = "mangalarapumanu@gmail.com,produde10053009@gmail.com"
    
    # Path to .env file
    env_path = '.env'
    
    # Check if the .env file exists
    if not os.path.exists(env_path):
        print(f"Creating new .env file at {os.path.abspath(env_path)}")
        new_file = True
    else:
        print(f"Updating existing .env file at {os.path.abspath(env_path)}")
        new_file = False
        
        # Read existing configuration
        current_config = {}
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        current_config[key] = value
            
            # Use existing app password if available
            if 'EMAIL_APP_PASSWORD' in current_config:
                app_password = current_config['EMAIL_APP_PASSWORD']
                
            # Keep existing emergency contacts if available
            if 'EMERGENCY_CONTACTS' in current_config:
                emergency_contacts = current_config['EMERGENCY_CONTACTS']
        except Exception as e:
            print(f"Error reading existing configuration: {e}")
            print("Will create new configuration.")
            new_file = True
    
    # Create the new configuration
    new_config = [
        f"EMAIL_APP_PASSWORD={app_password}",
        f"EMAIL_ADDRESS={correct_email}",
        f"EMERGENCY_CONTACTS={emergency_contacts}",
    ]
    
    # Write the configuration to the .env file
    try:
        with open(env_path, 'w') as f:
            for line in new_config:
                f.write(f"{line}\n")
        print(f"Successfully {'created' if new_file else 'updated'} .env file!")
        print("New configuration:")
        for line in new_config:
            if "PASSWORD" in line:
                print(f"{line.split('=')[0]}=******")
            else:
                print(line)
        return True
    except Exception as e:
        print(f"Error writing configuration: {e}")
        return False

if __name__ == "__main__":
    print("HAWKEYEZ Environment Configuration Fix Tool")
    print("==========================================")
    
    # Ask for confirmation if the .env file exists
    env_path = '.env'
    if os.path.exists(env_path):
        confirm = input("This will update your .env file. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    # Fix the .env file
    success = fix_env_file()
    
    if success:
        print("\nEnvironment configuration fixed successfully!")
        print("You can now run the application with: streamlit run app.py")
    else:
        print("\nFailed to fix environment configuration.")
        print("Please check the error message and try again.") 