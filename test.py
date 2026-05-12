from skill_agent import SkillAgent
                                                                                                                                    
agent = SkillAgent(db_path="skills.db")  # needs ANTHROPIC_API_KEY in env                                                         

# Ask a question                                                                                                                  
result = agent.run("Calculate compound interest on $5000 at 4.2% over 7 years")
print(result.answer)                                                                                                              
                                                            
# Give feedback                                                                                                                   
agent.feedback(result, positive=True)                     
agent.feedback(result, positive=False, comment="wrong formula")                                                                   
                                                                                                                                    
# Inspect the registry
tools = agent.registry.list_tools()                                                                                               
tool = agent.registry.get_tool_by_name("calculate_compound_interest")
versions = agent.registry.get_versions(tool.tool_id)                                                                              
                                                                                                                                    
# Prune bad/stale/duplicate tools                                                                                                 
retired = agent.registry.prune(stale_days=30)                                                                                     
                                                                                                                                    
# Manually retire a tool                                  
agent.registry.retire_tool(tool.tool_id)