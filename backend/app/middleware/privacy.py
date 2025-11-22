import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class PIISanitizer:
    """Sanitize personally identifiable information from text"""
    
    def __init__(self):
        # Comprehensive PII patterns - removes entire lines containing PII
        self.patterns = {
            'patient_info_section': [
                # Remove entire "Patient Information:" section
                (r'(?i)Patient Information:.*?(?=\n\n|\n[A-Z]|\Z)', ''),
            ],
            'name_line': [
                # Remove lines with names
                (r'(?i)^.*?(?:patient\s+name|name|patient)[\s:]+[A-Z][A-Z.\s]+.*?$', '', re.MULTILINE),
                (r'(?i)^.*?\b(?:MR|DR|MS|MRS|MISS)\.?\s+[A-Z][A-Z.\s]+.*?$', '', re.MULTILINE),
            ],
            'age_line': [
                # Remove lines with age
                (r'(?i)^.*?(?:age)[\s:]+\d{1,3}\s*(?:years?|yrs?)?.*?$', '', re.MULTILINE),
            ],
            'gender_line': [
                # Remove lines with gender
                (r'(?i)^.*?(?:gender|sex)[\s:]+\w+.*?$', '', re.MULTILINE),
            ],
            'location_line': [
                # Remove lines with location
                (r'(?i)^.*?(?:location|address|city|area)[\s:]+[A-Z][A-Z\s,]+.*?$', '', re.MULTILINE),
            ],
            'id_line': [
                # Remove lines with patient IDs
                (r'(?i)^.*?(?:patient\s+id|mrn|medical\s+record|uhid|id|patient\s+number)[\s:]+[A-Z0-9-]+.*?$', '', re.MULTILINE),
            ],
            'dob_line': [
                # Remove lines with DOB
                (r'(?i)^.*?(?:dob|d\.o\.b|date\s+of\s+birth|birth\s+date)[\s:]+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}.*?$', '', re.MULTILINE),
            ],
            'contact_line': [
                # Remove lines with phone/email
                (r'(?i)^.*?(?:phone|tel|mobile|contact|email|e-mail)[\s:]+.*?$', '', re.MULTILINE),
            ],
        }
    
    def sanitize(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Sanitize PII from text by removing entire lines containing PII
        Returns: (sanitized_text, redaction_stats)
        """
        if not text:
            return text, {}
        
        sanitized_text = text
        redaction_stats = {}
        
        for category, pattern_list in self.patterns.items():
            count = 0
            for item in pattern_list:
                pattern, replacement = item[0], item[1]
                flags = item[2] if len(item) == 3 else 0
                
                matches = re.findall(pattern, sanitized_text, flags=flags)
                if matches:
                    count += len(matches)
                    sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=flags)
            
            if count > 0:
                redaction_stats[category] = count
        
        # Clean up multiple consecutive newlines
        sanitized_text = re.sub(r'\n{3,}', '\n\n', sanitized_text)
        # Remove leading/trailing whitespace
        sanitized_text = sanitized_text.strip()
        
        if redaction_stats:
            logger.info(f"PII redacted (lines removed): {redaction_stats}")
        
        return sanitized_text, redaction_stats
    
    def sanitize_batch(self, texts: List[str]) -> List[Tuple[str, Dict[str, int]]]:
        """Sanitize multiple texts"""
        return [self.sanitize(text) for text in texts]


# Global sanitizer instance
sanitizer = PIISanitizer()


def sanitize_text(text: str) -> str:
    """Convenience function to sanitize text"""
    sanitized, _ = sanitizer.sanitize(text)
    return sanitized
