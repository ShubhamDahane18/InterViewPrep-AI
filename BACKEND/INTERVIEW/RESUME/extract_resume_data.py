# from BACKEND.INTERVIEW.RESUME.schema import ExtractResumeData
# from BACKEND.INTERVIEW.RESUME.state import ResumeAgentState
# from BACKEND.INTERVIEW.Utils.util import load_llm

# def extract_resume_data(state: ResumeAgentState) -> ResumeAgentState:
#     try:
#         # Load the structured-output LLM with the expected schema
#         llm = load_llm().with_structured_output(ExtractResumeData)
        
#         # Extract inputs from state
#         full_text = state.get('full_text' , '')
#         links = state.get('links' , [])

#         # Combine text and links into a single input string
#         resume_input = full_text + "\n\nLinks:\n" + "\n".join(
#         f"Page: {link['page']}, Text: {link['text']}, URL: {link['url']}" for link in links)

#         # Invoke the LLM to extract structured data
#         extracted_data = llm.invoke(resume_input)

#         # Return the updated state
#         return extracted_data.model_dump()

#     except Exception as e:
#         raise RuntimeError(f"Failed to extract resume data: {e}")

from BACKEND.INTERVIEW.RESUME.schema import ExtractResumeData
from BACKEND.INTERVIEW.RESUME.state import ResumeAgentState
from BACKEND.INTERVIEW.Utils.util import load_llm

def extract_resume_data(state: ResumeAgentState) -> ResumeAgentState:
    try:
        print("=== DEBUGGING extract_resume_data ===")
        print(f"Input state keys: {list(state.keys())}")
        
        # Load the structured-output LLM with the expected schema
        llm = load_llm().with_structured_output(ExtractResumeData)
        print("✅ LLM loaded successfully")
        
        # Extract inputs from state
        full_text = state.get('full_text', '')
        links = state.get('links', [])
        
        print(f"Full text length: {len(full_text)}")
        print(f"Links type: {type(links)}")
        print(f"Links content: {links}")
        
        if not full_text.strip():
            raise ValueError("No valid text content found in resume")
        
        # Handle links more safely
        links_text = ""
        if links:
            try:
                # Check the format of the first link to understand structure
                if isinstance(links, list) and len(links) > 0:
                    print(f"First link structure: {links[0]}")
                    print(f"First link type: {type(links[0])}")
                    
                    # Try to process links based on their actual structure
                    links_list = []
                    for i, link in enumerate(links):
                        try:
                            if isinstance(link, dict):
                                # Handle dictionary format
                                page = link.get('page', f'Page {i+1}')
                                text = link.get('text', '')
                                url = link.get('url', link.get('link', ''))
                                links_list.append(f"Page: {page}, Text: {text}, URL: {url}")
                            elif isinstance(link, str):
                                # Handle string format
                                links_list.append(f"Link: {link}")
                            else:
                                # Handle other formats
                                links_list.append(f"Link: {str(link)}")
                        except Exception as link_error:
                            print(f"Error processing link {i}: {link_error}")
                            links_list.append(f"Link {i}: {str(link)}")
                    
                    links_text = "\n".join(links_list)
            except Exception as links_error:
                print(f"Error processing links: {links_error}")
                links_text = f"Links: {str(links)}"
        
        # Combine text and links into a single input string
        if links_text:
            resume_input = full_text + "\n\nLinks:\n" + links_text
        else:
            resume_input = full_text
        
        print(f"Resume input length: {len(resume_input)}")
        print("🚀 Calling LLM...")
        
        # Invoke the LLM to extract structured data
        extracted_data = llm.invoke(resume_input)
        
        print("✅ LLM extraction completed")
        print(f"Extracted data type: {type(extracted_data)}")
        
        # Convert to dictionary
        extracted_dict = extracted_data.model_dump()
        print(f"Extracted fields: {list(extracted_dict.keys())}")
        
        # Create the return state by copying input state and updating with extracted data
        result_state = dict(state)  # Copy existing state
        result_state.update(extracted_dict)  # Add extracted data
        result_state['message'] = "Resume data extracted successfully"
        
        print("✅ State updated successfully")
        return result_state

    except Exception as e:
        print(f"❌ ERROR in extract_resume_data: {e}")
        print(f"❌ ERROR TYPE: {type(e).__name__}")
        
        import traceback
        print("📋 FULL TRACEBACK:")
        traceback.print_exc()
        
        # Return state with error instead of raising exception
        error_state = dict(state)
        error_state['message'] = f"Failed to extract resume data: {str(e)}"
        return error_state