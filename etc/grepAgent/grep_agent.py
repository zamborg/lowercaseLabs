"""
GrepAgent - An agent specialized in finding function definitions using bash commands.
"""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import weave

from core.base import Agent, LM, tool
from grepAgent.grep_environment import GrepTaskEnvironment


class GrepAgent(Agent):
    """
    Agent that uses bash commands (primarily grep) to find function definitions in repositories.
    """
    
    @weave.op()
    def __init__(self, lm: LM, environment: GrepTaskEnvironment):
        super().__init__(lm, environment)
        self.function_name = environment.get_state()["function_name"]
        self.search_patterns = self._generate_search_patterns(self.function_name)
        self.searched_patterns = set()
        self.potential_matches = []
    
    @weave.op()
    def _generate_search_patterns(self, function_name: str) -> List[str]:
        """Generate various patterns to search for function definitions."""
        patterns = []
        
        # Python patterns
        patterns.append(f"def {function_name}\\(")
        patterns.append(f"def {function_name} \\(")
        patterns.append(f"async def {function_name}\\(")
        patterns.append(f"async def {function_name} \\(")
        
        # JavaScript/TypeScript patterns  
        patterns.append(f"function {function_name}\\(")
        patterns.append(f"function {function_name} \\(")
        patterns.append(f"const {function_name} = ")
        patterns.append(f"let {function_name} = ")
        patterns.append(f"var {function_name} = ")
        patterns.append(f"{function_name}: function")
        patterns.append(f"export function {function_name}")
        patterns.append(f"export const {function_name}")
        
        # Java/C++/C# patterns
        patterns.append(f" {function_name}\\(")  # Space before for return type
        patterns.append(f"\\b{function_name}\\(")  # Word boundary
        
        # Ruby patterns
        patterns.append(f"def {function_name}")
        patterns.append(f"def self.{function_name}")
        
        # Go patterns
        patterns.append(f"func {function_name}\\(")
        patterns.append(f"func \\(.*\\) {function_name}\\(")  # Method
        
        # Class method patterns (various languages)
        patterns.append(f"\\.{function_name} = function")
        patterns.append(f"::{function_name}\\(")
        
        return patterns
    
    @weave.op()
    def step(self, environment: GrepTaskEnvironment) -> Dict[str, Any]:
        """Execute one step of the search process."""
        state = self.ingest_state(environment)
        
        # Check if we've already found a confident match
        if self.potential_matches and self._should_submit_answer():
            best_match = self._select_best_match()
            return self._submit_final_answer(best_match)
        
        # Determine next search strategy
        messages = self._build_search_strategy_prompt(state)
        result = self.invoke(messages)
        
        return {
            "status": "searching",
            "patterns_tried": len(self.searched_patterns),
            "matches_found": len(self.potential_matches)
        }
    
    @weave.op()
    def _build_search_strategy_prompt(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build prompt for determining next search action."""
        system_prompt = f"""You are GrepAgent, specialized in finding function definitions in code repositories.

You are searching for the function: {self.function_name}

Available tools:
- run_command: Execute any bash command in the repository
- check_file: Read a specific file's content
- submit_answer: Submit the final answer when confident

Search strategies:
1. Start with broad grep searches across common patterns
2. Use find to locate relevant file types first
3. Narrow down with more specific searches
4. Verify matches by checking file content

Current repository: {state['repo_path']}
Commands executed so far: {len(state['commands_executed'])}
Potential matches found: {len(self.potential_matches)}"""
        
        # Build context from previous searches
        recent_commands = state['commands_executed'][-5:] if state['commands_executed'] else []
        context = "\nRecent commands:\n"
        for cmd in recent_commands:
            context += f"$ {cmd['command']}\n"
            if cmd.get('stdout_lines', 0) > 0:
                context += f"  (Found {cmd['stdout_lines']} lines)\n"
        
        user_prompt = f"Find the definition of function '{self.function_name}'.{context}\n\nWhat should we search for next?"
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @weave.op()
    @tool
    def run_command(self, command: str) -> str:
        """
        Run a bash command in the repository.
        
        Args:
            command: Bash command to execute
        """
        env = self.environment
        success, stdout, stderr = env.execute_command(command)
        
        # Parse grep results if it's a grep command
        if "grep" in command and success and stdout:
            self._parse_grep_results(stdout, command)
        
        if success:
            return f"Command output:\n{stdout}" if stdout else "Command completed (no output)"
        else:
            return f"Command failed:\n{stderr}" if stderr else "Command failed (no error message)"
    
    @weave.op()
    @tool 
    def check_file(self, filepath: str, start_line: int = 1, end_line: int = 50) -> str:
        """
        Check content of a specific file.
        
        Args:
            filepath: Path to file relative to repository root
            start_line: Starting line number
            end_line: Ending line number
        """
        env = self.environment
        content = env.get_file_content(filepath)
        
        if content is None:
            return f"Error: File '{filepath}' not found"
        
        lines = content.splitlines()
        selected_lines = lines[start_line-1:end_line]
        
        # Format with line numbers
        formatted = []
        for i, line in enumerate(selected_lines, start=start_line):
            formatted.append(f"{i:4d}: {line}")
        
        # Check if function is defined in this section
        content_section = "\n".join(selected_lines)
        if self.function_name in content_section:
            # Look for actual definition
            for i, line in enumerate(selected_lines, start=start_line):
                if self._is_function_definition(line, self.function_name):
                    self.potential_matches.append({
                        "filepath": filepath,
                        "line_number": i,
                        "line_content": line.strip(),
                        "confidence": 0.9
                    })
        
        return f"Content of {filepath} (lines {start_line}-{end_line}):\n" + "\n".join(formatted)
    
    @weave.op()
    @tool
    def submit_answer(self, filepath: str, line_number: int = None) -> str:
        """
        Submit the final answer for where the function is defined.
        
        Args:
            filepath: Path to the file containing the function
            line_number: Optional line number where function starts
        """
        env = self.environment
        env.submit_answer(filepath, line_number)
        
        return f"Submitted answer: {self.function_name} is defined in {filepath}" + \
               (f" at line {line_number}" if line_number else "")
    
    @weave.op()
    def _parse_grep_results(self, grep_output: str, command: str):
        """Parse grep output to find potential matches."""
        lines = grep_output.strip().split('\n')
        
        for line in lines:
            if not line:
                continue
                
            # Try to parse different grep output formats
            # Format: filename:line_number:content
            match = re.match(r'^([^:]+):(\d+):(.*)$', line)
            if match:
                filepath, line_num, content = match.groups()
                
                # Check if this looks like a function definition
                if self._is_function_definition(content, self.function_name):
                    self.potential_matches.append({
                        "filepath": filepath,
                        "line_number": int(line_num),
                        "line_content": content.strip(),
                        "confidence": self._calculate_confidence(content, filepath)
                    })
            
            # Format: filename:content (no line numbers)
            elif ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    filepath, content = parts
                    if self._is_function_definition(content, self.function_name):
                        self.potential_matches.append({
                            "filepath": filepath,
                            "line_number": None,
                            "line_content": content.strip(),
                            "confidence": self._calculate_confidence(content, filepath) * 0.8
                        })
    
    @weave.op()
    def _is_function_definition(self, line: str, function_name: str) -> bool:
        """Check if a line likely contains a function definition."""
        line = line.strip()
        
        # Skip comments and strings
        if line.startswith('#') or line.startswith('//') or line.startswith('/*'):
            return False
        
        # Common patterns
        patterns = [
            f"def {function_name}",
            f"function {function_name}",
            f"func {function_name}",
            f"{function_name}:",
            f"{function_name} =",
            f"{function_name}(",
        ]
        
        return any(pattern in line for pattern in patterns)
    
    @weave.op()
    def _calculate_confidence(self, content: str, filepath: str) -> float:
        """Calculate confidence score for a match."""
        confidence = 0.5
        
        # Boost for definition keywords
        if any(keyword in content for keyword in ['def ', 'function ', 'func ']):
            confidence += 0.3
        
        # Boost for being in likely source files
        if any(filepath.endswith(ext) for ext in ['.py', '.js', '.ts', '.go', '.rb', '.java']):
            confidence += 0.1
        
        # Reduce for test files
        if 'test' in filepath.lower():
            confidence *= 0.7
        
        # Boost for exact name match with proper boundaries
        if re.search(rf'\b{self.function_name}\b', content):
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    @weave.op()
    def _should_submit_answer(self) -> bool:
        """Determine if we should submit an answer based on current matches."""
        if not self.potential_matches:
            return False
        
        # Submit if we have a high confidence match
        best_match = max(self.potential_matches, key=lambda x: x['confidence'])
        if best_match['confidence'] >= 0.8:
            return True
        
        # Submit if we've searched enough
        state = self.environment.get_state()
        if len(state['commands_executed']) > 15:
            return True
            
        return False
    
    @weave.op()
    def _select_best_match(self) -> Dict[str, Any]:
        """Select the best match from potential matches."""
        # Sort by confidence, preferring matches with line numbers
        sorted_matches = sorted(
            self.potential_matches,
            key=lambda x: (x['confidence'], x['line_number'] is not None),
            reverse=True
        )
        
        return sorted_matches[0]
    
    @weave.op()
    def _submit_final_answer(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Submit the final answer and return completion status."""
        self.environment.submit_answer(match['filepath'], match.get('line_number'))
        
        return {
            "status": "completed",
            "answer": match['filepath'],
            "line_number": match.get('line_number'),
            "confidence": match['confidence']
        }