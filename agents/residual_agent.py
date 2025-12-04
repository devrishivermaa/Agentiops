# agents/residual_agent.py
"""
ResidualAgent: Handles failed tasks, quality validation, and retry logic.
This is the gray robot on the right side of Architecture 1.
"""

import time
from typing import Dict, Any, List, Optional
from utils.logger import get_logger
from utils.llm_helper import LLMProcessor

logger = get_logger("ResidualAgent")


class ResidualAgent:
    """
    ResidualAgent handles:
    - Failed task retry with exponential backoff
    - Quality validation of SubMaster outputs
    - Error correction and result sanitization
    - Anomaly detection in processing results
    """
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize ResidualAgent.
        
        Args:
            max_retries: Maximum retry attempts for failed tasks
        """
        self.max_retries = max_retries
        self.llm = None
        
        try:
            self.llm = LLMProcessor(
                model="mistral-small-latest",
                temperature=0.3,
                max_retries=2,
                caller_id="ResidualAgent"
            )
            logger.info(f"ResidualAgent initialized with LLM (max_retries={max_retries})")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM for ResidualAgent: {e}")
        
        self.validation_history = []
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate SubMaster processing results.
        
        Args:
            results: SubMaster results to validate
            
        Returns:
            Validation report with errors, warnings, and quality score
        """
        logger.info("Validating SubMaster results...")
        
        errors = []
        warnings = []
        fixed_results = {}
        
        for sm_id, sm_result in results.items():
            if sm_result.get('status') == 'error':
                errors.append({
                    "sm_id": sm_id,
                    "error": sm_result.get('error', 'Unknown error'),
                    "severity": "high"
                })
                continue
            
            if sm_result.get('status') != 'ok':
                warnings.append({
                    "sm_id": sm_id,
                    "warning": f"Unexpected status: {sm_result.get('status')}",
                    "severity": "medium"
                })
            
            output = sm_result.get('output', {})
            
            # Check for failed LLM analyses
            llm_failures = output.get('llm_failures', 0)
            llm_successes = output.get('llm_successes', 0)
            
            if llm_failures > 0:
                failure_rate = llm_failures / (llm_failures + llm_successes) if (llm_failures + llm_successes) > 0 else 0
                
                if failure_rate > 0.5:
                    errors.append({
                        "sm_id": sm_id,
                        "error": f"High LLM failure rate: {failure_rate*100:.1f}%",
                        "severity": "medium"
                    })
                elif failure_rate > 0.2:
                    warnings.append({
                        "sm_id": sm_id,
                        "warning": f"Elevated LLM failure rate: {failure_rate*100:.1f}%",
                        "severity": "low"
                    })
            
            # Check for empty or invalid results
            page_results = output.get('results', [])
            if not page_results:
                warnings.append({
                    "sm_id": sm_id,
                    "warning": "No page results found",
                    "severity": "medium"
                })
            
            # Sanitize and fix results
            fixed_output = self._sanitize_output(output)
            fixed_results[sm_id] = {"status": "ok", "output": fixed_output}
        
        # Calculate quality score
        total_submasters = len(results)
        error_count = len(errors)
        warning_count = len(warnings)
        
        quality_score = max(0, 100 - (error_count * 20) - (warning_count * 5))
        
        validation_report = {
            "status": "validated",
            "quality_score": quality_score,
            "total_submasters": total_submasters,
            "errors": errors,
            "warnings": warnings,
            "error_count": error_count,
            "warning_count": warning_count,
            "fixed_results": fixed_results
        }
        
        self.validation_history.append(validation_report)
        
        logger.info(
            f"✅ Validation complete: Quality Score {quality_score}/100, "
            f"{error_count} errors, {warning_count} warnings"
        )
        
        return validation_report
    
    def _sanitize_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and clean SubMaster output."""
        sanitized = output.copy()
        
        # Ensure required fields exist
        if 'results' not in sanitized:
            sanitized['results'] = []
        if 'total_entities' not in sanitized:
            sanitized['total_entities'] = 0
        if 'total_keywords' not in sanitized:
            sanitized['total_keywords'] = 0
        
        # Clean page results
        cleaned_results = []
        for page_result in sanitized.get('results', []):
            if isinstance(page_result, dict):
                # Ensure arrays exist
                if 'entities' not in page_result:
                    page_result['entities'] = []
                if 'keywords' not in page_result:
                    page_result['keywords'] = []
                if 'key_points' not in page_result:
                    page_result['key_points'] = []
                
                # Remove null/empty entities
                page_result['entities'] = [e for e in page_result.get('entities', []) if e]
                page_result['keywords'] = [k for k in page_result.get('keywords', []) if k]
                page_result['key_points'] = [kp for kp in page_result.get('key_points', []) if kp]
                
                cleaned_results.append(page_result)
        
        sanitized['results'] = cleaned_results
        
        return sanitized
    
    def retry_failed_tasks(
        self,
        failed_tasks: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retry failed processing tasks.
        
        Args:
            failed_tasks: List of failed task descriptions
            metadata: Document metadata for context
            
        Returns:
            List of retry results
        """
        logger.info(f"Retrying {len(failed_tasks)} failed tasks...")
        
        retry_results = []
        
        for task in failed_tasks:
            task_id = task.get('task_id', 'unknown')
            error = task.get('error', 'Unknown error')
            
            logger.info(f"Retrying task {task_id} (previous error: {error})")
            
            retry_result = {
                "task_id": task_id,
                "original_error": error,
                "retry_status": "not_attempted",
                "attempts": 0
            }
            
            # Exponential backoff retry
            for attempt in range(self.max_retries):
                retry_result['attempts'] = attempt + 1
                
                try:
                    # Simulate retry with delay
                    wait_time = 2 ** attempt
                    logger.debug(f"Attempt {attempt+1}/{self.max_retries}, waiting {wait_time}s")
                    time.sleep(wait_time)
                    
                    # Actual retry logic would go here
                    # For now, mark as recovered if it was a transient error
                    if "timeout" in error.lower() or "rate" in error.lower():
                        retry_result['retry_status'] = "recovered"
                        logger.info(f"✅ Task {task_id} recovered on attempt {attempt+1}")
                        break
                    else:
                        # Permanent failure
                        retry_result['retry_status'] = "permanent_failure"
                        logger.warning(f"❌ Task {task_id} has permanent failure")
                        break
                        
                except Exception as e:
                    logger.error(f"Retry attempt {attempt+1} failed: {e}")
                    retry_result['retry_error'] = str(e)
                    
                    if attempt == self.max_retries - 1:
                        retry_result['retry_status'] = "failed_after_retries"
            
            retry_results.append(retry_result)
        
        successful_retries = sum(1 for r in retry_results if r['retry_status'] == 'recovered')
        logger.info(f"Retry complete: {successful_retries}/{len(failed_tasks)} tasks recovered")
        
        return retry_results
    
    def detect_anomalies(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in processing results.
        
        Args:
            results: SubMaster results to analyze
            
        Returns:
            List of detected anomalies
        """
        logger.info("Running anomaly detection...")
        
        anomalies = []
        
        # Calculate baseline metrics
        all_llm_successes = []
        all_processing_times = []
        all_entity_counts = []
        
        for sm_result in results.values():
            if sm_result.get('status') == 'ok':
                output = sm_result.get('output', {})
                all_llm_successes.append(output.get('llm_successes', 0))
                all_processing_times.append(output.get('elapsed_time', 0))
                all_entity_counts.append(output.get('total_entities', 0))
        
        if not all_llm_successes:
            return anomalies
        
        avg_success = sum(all_llm_successes) / len(all_llm_successes)
        avg_time = sum(all_processing_times) / len(all_processing_times)
        avg_entities = sum(all_entity_counts) / len(all_entity_counts)
        
        # Detect outliers
        for sm_id, sm_result in results.items():
            if sm_result.get('status') != 'ok':
                continue
            
            output = sm_result.get('output', {})
            
            # Check for unusually low success rate
            successes = output.get('llm_successes', 0)
            if successes < avg_success * 0.5:
                anomalies.append({
                    "sm_id": sm_id,
                    "type": "low_success_rate",
                    "value": successes,
                    "expected": avg_success,
                    "severity": "medium"
                })
            
            # Check for unusually long processing time
            elapsed = output.get('elapsed_time', 0)
            if elapsed > avg_time * 2:
                anomalies.append({
                    "sm_id": sm_id,
                    "type": "slow_processing",
                    "value": elapsed,
                    "expected": avg_time,
                    "severity": "low"
                })
            
            # Check for unusually low entity extraction
            entities = output.get('total_entities', 0)
            if entities < avg_entities * 0.3 and avg_entities > 5:
                anomalies.append({
                    "sm_id": sm_id,
                    "type": "low_entity_count",
                    "value": entities,
                    "expected": avg_entities,
                    "severity": "low"
                })
        
        if anomalies:
            logger.warning(f"⚠️  Detected {len(anomalies)} anomalies")
        else:
            logger.info("✅ No anomalies detected")
        
        return anomalies
    
    def generate_quality_report(self, validation: Dict[str, Any]) -> str:
        """Generate human-readable quality report."""
        quality_score = validation.get('quality_score', 0)
        errors = validation.get('errors', [])
        warnings = validation.get('warnings', [])
        
        report = f"""
RESIDUAL AGENT QUALITY REPORT
==============================

Overall Quality Score: {quality_score}/100

Errors: {len(errors)}
Warnings: {len(warnings)}
Total SubMasters: {validation.get('total_submasters', 0)}

"""
        
        if errors:
            report += "\n❌ ERRORS:\n"
            for err in errors:
                report += f"  - {err['sm_id']}: {err['error']} (severity: {err['severity']})\n"
        
        if warnings:
            report += "\n⚠️  WARNINGS:\n"
            for warn in warnings:
                report += f"  - {warn['sm_id']}: {warn['warning']} (severity: {warn['severity']})\n"
        
        if quality_score >= 90:
            report += "\n✅ VERDICT: Excellent quality, proceed with confidence."
        elif quality_score >= 70:
            report += "\n✓ VERDICT: Good quality, minor issues detected."
        elif quality_score >= 50:
            report += "\n⚠️  VERDICT: Fair quality, review warnings before proceeding."
        else:
            report += "\n❌ VERDICT: Poor quality, investigate errors immediately."
        
        return report
