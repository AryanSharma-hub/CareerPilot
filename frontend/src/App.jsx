import { useState, useCallback, useRef } from "react";
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import jsPDF from "jspdf";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function getSessionUserId() {
  const key = "careerpilot_user_id";
  let id = sessionStorage.getItem(key);
  if (!id) { id = "user_" + Math.random().toString(36).slice(2,11) + "_" + Date.now(); sessionStorage.setItem(key, id); }
  return id;
}
const SESSION_USER_ID = getSessionUserId();
const SCORE_COLORS = ["#6366f1","#3b82f6","#10b981","#f59e0b","#8b5cf6"];
const SCORE_LABELS = ["Experience","Skills","Impact","Projects","Formatting"];

function ScoreBadge({ score }) {
  const c = score>=80?"bg-emerald-50 text-emerald-700 border-emerald-200":score>=60?"bg-amber-50 text-amber-700 border-amber-200":"bg-red-50 text-red-700 border-red-200";
  const l = score>=80?"Strong":score>=60?"Moderate":"Weak";
  return <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${c}`}>{l}</span>;
}
function Tag({ label, variant="default" }) {
  const s = {default:"bg-slate-100 text-slate-700",blue:"bg-blue-50 text-blue-700",green:"bg-emerald-50 text-emerald-700",red:"bg-red-50 text-red-700",amber:"bg-amber-50 text-amber-700",purple:"bg-purple-50 text-purple-700",indigo:"bg-indigo-50 text-indigo-700",yellow:"bg-yellow-50 text-yellow-700"};
  return <span className={`inline-block px-2.5 py-1 rounded-md text-xs font-medium ${s[variant]||s.default}`}>{label}</span>;
}
function Card({ children, className="" }) { return <div className={`bg-white border border-slate-200 rounded-2xl p-6 ${className}`}>{children}</div>; }
function SectionHeading({ children }) { return <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">{children}</h3>; }
function ErrorBanner({ message }) {
  if(!message) return null;
  return <div className="flex items-start gap-2 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl mt-3 text-sm"><svg className="w-4 h-4 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>{message}</div>;
}
function Spinner({ label }) {
  return <span className="flex items-center justify-center gap-2"><svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/></svg>{label}</span>;
}
function PrimaryBtn({ onClick, disabled, loading, loadingLabel, children, className="" }) {
  return <button onClick={onClick} disabled={disabled||loading} className={`w-full py-3 px-4 rounded-xl font-medium text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed ${className}`}>{loading?<Spinner label={loadingLabel}/>:children}</button>;
}
function EmptyState({ icon, message }) { return <div className="flex flex-col items-center justify-center py-6 text-slate-400"><div className="text-2xl mb-2">{icon}</div><p className="text-sm">{message}</p></div>; }
function StepIndicator({ steps, current }) {
  return <div className="flex items-center gap-2 mb-8">{steps.map((s,i)=>(
    <div key={i} className="flex items-center gap-2">
      <div className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-semibold transition-colors ${i<current?"bg-indigo-600 text-white":i===current?"bg-indigo-100 text-indigo-700 border-2 border-indigo-600":"bg-slate-100 text-slate-400"}`}>
        {i<current?<svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7"/></svg>:i+1}
      </div>
      <span className={`text-xs font-medium hidden sm:block ${i===current?"text-slate-800":"text-slate-400"}`}>{s}</span>
      {i<steps.length-1&&<div className={`w-6 h-px mx-1 ${i<current?"bg-indigo-600":"bg-slate-200"}`}/>}
    </div>
  ))}</div>;
}

const STEPS = ["Upload","Analyze","Match","Prepare"];

export default function App() {
  const [file,setFile]=useState(null);
  const [loading,setLoading]=useState(false);
  const [uploadError,setUploadError]=useState("");
  const [result,setResult]=useState(null);
  const [jobDescription,setJobDescription]=useState("");
  const [jobMatchLoading,setJobMatchLoading]=useState(false);
  const [jobMatchError,setJobMatchError]=useState("");
  const [jobMatchResult,setJobMatchResult]=useState(null);
  const [targetRole,setTargetRole]=useState("");
  const [rewriteLoading,setRewriteLoading]=useState(false);
  const [rewriteError,setRewriteError]=useState("");
  const [rewriteResult,setRewriteResult]=useState(null);
  const [chatMessage,setChatMessage]=useState("");
  const [chatLoading,setChatLoading]=useState(false);
  const [chatError,setChatError]=useState("");
  const [chatHistory,setChatHistory]=useState([]);
  const [interviewRole,setInterviewRole]=useState("");
  const [interviewLoading,setInterviewLoading]=useState(false);
  const [interviewError,setInterviewError]=useState("");
  const [interviewQuestions,setInterviewQuestions]=useState(null);
  const [activeTab,setActiveTab]=useState("analysis");
  const chatEndRef=useRef(null);

  const currentStep=!result?0:!jobMatchResult?1:activeTab==="interview"||activeTab==="chat"?3:2;
  const scoreData=result?SCORE_LABELS.map((name,i)=>({name,score:[result.llm_analysis?.score_breakdown?.experience_score,result.llm_analysis?.score_breakdown?.skills_score,result.llm_analysis?.score_breakdown?.impact_score,result.llm_analysis?.score_breakdown?.project_score,result.llm_analysis?.score_breakdown?.formatting_score][i]||0})):[];

  const handleFileChange=useCallback((e)=>{
    const f=e.target.files[0]; setUploadError("");
    if(!f)return;
    if(!f.name.toLowerCase().endsWith(".pdf")){setUploadError("Only PDF files are supported.");return;}
    setFile(f);
  },[]);

  const handleUpload=useCallback(async()=>{
    if(!file){setUploadError("Please select a PDF file.");return;}
    const fd=new FormData(); fd.append("file",file);
    try{setLoading(true);setUploadError("");setResult(null);setJobMatchResult(null);setRewriteResult(null);setInterviewQuestions(null);setChatHistory([]);
      const res=await axios.post(`${API_BASE}/upload-resume`,fd,{headers:{"Content-Type":"multipart/form-data"}});
      setResult(res.data);setActiveTab("analysis");
    }catch(e){setUploadError(e.response?.data?.detail||"Analysis failed. Please try again.");}
    finally{setLoading(false);}
  },[file]);

  const handleJobMatch=useCallback(async()=>{
    if(!jobDescription.trim()){setJobMatchError("Please enter a job description.");return;}
    try{setJobMatchLoading(true);setJobMatchError("");
      const res=await axios.post(`${API_BASE}/match-job`,{resume_text:result.extracted_text,job_description:jobDescription});
      setJobMatchResult(res.data);setActiveTab("match");
    }catch(e){setJobMatchError(e.response?.data?.detail||"Job match failed.");}
    finally{setJobMatchLoading(false);}
  },[jobDescription,result]);

  const handleResumeRewrite=useCallback(async()=>{
    if(!targetRole.trim()){setRewriteError("Please enter a target role.");return;}
    try{setRewriteLoading(true);setRewriteError("");
      const res=await axios.post(`${API_BASE}/rewrite-resume`,{resume_text:result.extracted_text,target_role:targetRole});
      setRewriteResult(res.data);
    }catch(e){setRewriteError(e.response?.data?.detail||"Rewrite failed.");}
    finally{setRewriteLoading(false);}
  },[targetRole,result]);

  const handleCareerChat=useCallback(async()=>{
    if(!chatMessage.trim())return;
    const msg=chatMessage.trim();
    setChatHistory(p=>[...p,{role:"user",text:msg}]);setChatMessage("");
    try{setChatLoading(true);setChatError("");
      const res=await axios.post(`${API_BASE}/career-chat`,{user_id:SESSION_USER_ID,resume_analysis:jobMatchResult?.resume_analysis||{...result?.structured_resume_data,domain:result?.detected_domain?.domain}||{},ats_analysis:result?.llm_analysis||{},job_match_analysis:jobMatchResult?.job_match_analysis||{},user_question:msg});
      setChatHistory(p=>[...p,{role:"ai",text:res.data.career_response||"No response."}]);
      setTimeout(()=>chatEndRef.current?.scrollIntoView({behavior:"smooth"}),100);
    }catch(e){const err=e.response?.data?.detail||"Chat failed.";setChatError(err);setChatHistory(p=>[...p,{role:"error",text:err}]);}
    finally{setChatLoading(false);}
  },[chatMessage,result,jobMatchResult]);

  const handleInterviewGenerator=useCallback(async()=>{
    if(!interviewRole.trim()){setInterviewError("Please enter a target role.");return;}
    try{setInterviewLoading(true);setInterviewError("");
      const res=await axios.post(`${API_BASE}/generate-interview`,{resume_analysis:result?.structured_resume_data||{},job_match_analysis:jobMatchResult?.job_match_analysis||{},target_role:interviewRole});
      setInterviewQuestions(res.data);
    }catch(e){setInterviewError(e.response?.data?.detail||"Generation failed.");}
    finally{setInterviewLoading(false);}
  },[interviewRole,result,jobMatchResult]);

  const handleDownloadPDF=useCallback(()=>{
    if(!result)return;
    const doc=new jsPDF(); let y=20; const pw=doc.internal.pageSize.width;
    const chk=()=>{if(y>270){doc.addPage();y=20;}};
    const hdr=(t)=>{chk();doc.setFillColor(79,70,229);doc.rect(15,y-6,180,10,"F");doc.setTextColor(255,255,255);doc.setFontSize(11);doc.setFont("helvetica","bold");doc.text(t,20,y);doc.setTextColor(0,0,0);y+=12;};
    const ln=(t)=>{chk();doc.setFontSize(10);doc.setFont("helvetica","normal");const lines=doc.splitTextToSize(String(t||""),170);doc.text(lines,20,y);y+=lines.length*6;};
    doc.setFont("helvetica","bold");doc.setFontSize(20);doc.text("CareerPilot AI Report",pw/2,y,{align:"center"});y+=8;
    doc.setDrawColor(79,70,229);doc.line(15,y,195,y);y+=8;
    doc.setFontSize(9);doc.setFont("helvetica","normal");doc.setTextColor(120,120,120);
    ln(`Generated: ${new Date().toLocaleString()}`);ln(`Resume: ${result?.filename}`);doc.setTextColor(0,0,0);y+=4;

    hdr("ATS SCORE & DOMAIN");
    doc.setFontSize(24);doc.setFont("helvetica","bold");doc.text(`${result?.llm_analysis?.ats_score??"-"}/100`,20,y);y+=12;
    doc.setFontSize(10);doc.setFont("helvetica","normal");
    ln(`Domain: ${result?.detected_domain?.domain}`);y+=2;

    hdr("SCORE BREAKDOWN");
    const bd=result?.llm_analysis?.score_breakdown||{};
    [["Experience",bd.experience_score],["Skills",bd.skills_score],["Impact",bd.impact_score],["Projects",bd.project_score],["Formatting",bd.formatting_score]].forEach(([l,v])=>ln(`• ${l}: ${v??"-"}/100`));y+=3;

    hdr("TECHNICAL SKILLS");
    ln((result?.structured_resume_data?.technical_skills||[]).join(", ")||"None detected");y+=2;
    hdr("TOOLS & PLATFORMS");
    ln((result?.structured_resume_data?.tools||[]).join(", ")||"None detected");y+=2;
    hdr("SOFT SKILLS");
    ln((result?.structured_resume_data?.soft_skills||[]).join(", ")||"None detected");y+=2;

    hdr("RESUME STRENGTHS");
    (result?.llm_analysis?.strengths||[]).forEach(s=>ln(`• ${s}`));y+=2;
    hdr("AREAS TO IMPROVE");
    (result?.llm_analysis?.weaknesses||[]).forEach(w=>ln(`• ${w}`));y+=2;
    hdr("ATS SUGGESTIONS");
    (result?.llm_analysis?.suggestions||[]).forEach(s=>ln(`• ${s}`));y+=2;

    if(jobMatchResult){
      hdr("JOB MATCH RESULTS");
      ln(`Match Score: ${jobMatchResult?.job_match_analysis?.match_percentage}%`);
      ln(`Domain Alignment: ${jobMatchResult?.job_match_analysis?.domain_alignment}`);y+=2;

      ln("✓ Matched Skills:");
      (jobMatchResult?.job_match_analysis?.matched_skills||[]).forEach(s=>ln(`  • ${s}`));
      ln("≈ Related Skills:");
      (jobMatchResult?.job_match_analysis?.related_skills||[]).forEach(r=>{
        const detail = r.reason || (r.related_to!=="inferred from experience" ? `related to ${r.related_to}` : "");
        ln(`  • ${r.skill}${detail?` — ${detail}`:""}`);
      });
      ln("✗ Missing Skills:");
      (jobMatchResult?.job_match_analysis?.missing_skills||[]).forEach(s=>ln(`  • ${s}`));y+=2;

      const ins=jobMatchResult?.match_insights;
      if(ins){
        hdr("RECRUITER ASSESSMENT");
        ln(ins.summary||"");y+=2;
        ln("Strengths:");(ins.strengths||[]).forEach(s=>ln(`  • ${s}`));
        ln("Concerns:");(ins.concerns||[]).forEach(c=>ln(`  • ${c}`));
        ln(`Overall: ${ins.recommendation||""}`);y+=2;
      }
    }

    if(rewriteResult){
      hdr("OPTIMIZED RESUME BULLETS");
      (rewriteResult?.rewritten_resume?.optimized_bullets||[]).forEach(b=>ln(`• ${b}`));y+=2;
    }

    if(interviewQuestions){
      const cats=[
        ["BEHAVIORAL QUESTIONS","behavioral_questions"],
        ["TECHNICAL QUESTIONS","technical_questions"],
        ["LEADERSHIP QUESTIONS","leadership_questions"],
        ["SKILL GAP QUESTIONS","skill_gap_questions"],
        ["ROLE-SPECIFIC QUESTIONS","role_specific_questions"],
      ];
      cats.forEach(([label,key])=>{
        const qs=interviewQuestions?.interview_questions?.[key]||[];
        if(!qs.length)return;
        hdr(label);qs.forEach((q,i)=>ln(`Q${i+1}. ${q}`));y+=2;
      });
    }

    const tp=doc.getNumberOfPages();
    for(let i=1;i<=tp;i++){doc.setPage(i);doc.setFontSize(9);doc.setTextColor(160,160,160);doc.text(`CareerPilot AI  |  Page ${i} of ${tp}`,20,290);}
    doc.save("CareerPilot_Report.pdf");
  },[result,jobMatchResult,rewriteResult,interviewQuestions]);

  const ats=result?.llm_analysis?.ats_score;
  const matchPct=jobMatchResult?.job_match_analysis?.match_percentage;
  const tabs=[
    {key:"analysis",label:"Resume Analysis",show:!!result},
    {key:"match",label:"Job Match",show:!!result},
    {key:"rewrite",label:"Rewrite",show:!!result},
    {key:"chat",label:"AI Mentor",show:!!result},
    {key:"interview",label:"Interview Prep",show:!!result},
  ];

  return (
    <div className="min-h-screen bg-slate-50">
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 bg-indigo-600 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            </div>
            <span className="font-semibold text-slate-900 text-sm">CareerPilot</span>
            <span className="text-slate-300 text-xs hidden sm:block">AI Resume Intelligence</span>
          </div>
          {result&&<button onClick={handleDownloadPDF} className="flex items-center gap-1.5 text-xs font-medium text-indigo-600 hover:text-indigo-800 transition-colors"><svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>Download Report</button>}
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 py-8">
        <StepIndicator steps={STEPS} current={currentStep}/>

        {!result?(
          <div className="max-w-xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-slate-900 mb-2">Analyze your resume</h1>
              <p className="text-slate-500 text-sm">Upload your PDF resume for instant ATS scoring, skill analysis, and career insights.</p>
            </div>
            <Card>
              <label className="block w-full border-2 border-dashed border-slate-200 rounded-xl p-8 text-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-all group">
                <input type="file" accept=".pdf" onChange={handleFileChange} className="hidden"/>
                <div className="w-10 h-10 bg-slate-100 group-hover:bg-indigo-100 rounded-xl flex items-center justify-center mx-auto mb-3 transition-colors">
                  <svg className="w-5 h-5 text-slate-400 group-hover:text-indigo-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                </div>
                {file?<p className="text-sm font-medium text-indigo-600">{file.name}</p>:<><p className="text-sm font-medium text-slate-700">Drop your resume here or click to browse</p><p className="text-xs text-slate-400 mt-1">PDF only · Max 10MB</p></>}
              </label>
              <ErrorBanner message={uploadError}/>
              <PrimaryBtn onClick={handleUpload} loading={loading} loadingLabel="Analyzing your resume..." className="mt-4 bg-indigo-600 hover:bg-indigo-700 text-white">Analyze Resume →</PrimaryBtn>
            </Card>
          </div>
        ):(
          <>
            {/* Score banner */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
              {[
                {label:"ATS Score",value:`${ats}/100`,sub:<ScoreBadge score={ats}/>},
                {label:"Domain",value:result.detected_domain?.domain,sub:null},
                {label:"Technical Skills",value:result.structured_resume_data?.technical_skills?.length||0,sub:<span className="text-xs text-slate-400">detected</span>},
                {label:"Job Match",value:matchPct!=null?`${matchPct}%`:"—",sub:matchPct!=null?<ScoreBadge score={matchPct}/>:<span className="text-xs text-slate-400">run job match</span>},
              ].map((m,i)=>(
                <div key={i} className="bg-white border border-slate-200 rounded-2xl p-4">
                  <p className="text-xs text-slate-500 font-medium mb-1">{m.label}</p>
                  <p className="text-xl font-bold text-slate-900 leading-tight">{m.value}</p>
                  {m.sub&&<div className="mt-1">{m.sub}</div>}
                </div>
              ))}
            </div>

            {/* Tabs */}
            <div className="flex gap-1 bg-slate-100 p-1 rounded-xl mb-6 overflow-x-auto">
              {tabs.filter(t=>t.show).map(t=>(
                <button key={t.key} onClick={()=>setActiveTab(t.key)} className={`flex-1 min-w-max px-4 py-2 rounded-lg text-xs font-semibold transition-all ${activeTab===t.key?"bg-white text-indigo-700 shadow-sm":"text-slate-500 hover:text-slate-700"}`}>{t.label}</button>
              ))}
            </div>

            {/* ── ANALYSIS TAB ── */}
            {activeTab==="analysis"&&(
              <div className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <SectionHeading>Score Breakdown</SectionHeading>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={scoreData} margin={{top:4,right:4,bottom:4,left:-20}}>
                        <XAxis dataKey="name" tick={{fontSize:11,fill:"#94a3b8"}} axisLine={false} tickLine={false}/>
                        <YAxis domain={[0,100]} tick={{fontSize:11,fill:"#94a3b8"}} axisLine={false} tickLine={false}/>
                        <Tooltip formatter={(v)=>[`${v}/100`]} contentStyle={{border:"1px solid #e2e8f0",borderRadius:"12px",fontSize:"12px"}}/>
                        <Bar dataKey="score" radius={[6,6,0,0]}>{scoreData.map((_,i)=><Cell key={i} fill={SCORE_COLORS[i]}/>)}</Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                  <Card>
                    <SectionHeading>Strengths</SectionHeading>
                    {(result.llm_analysis?.strengths||[]).length>0
                      ?<ul className="space-y-2.5">{result.llm_analysis.strengths.map((s,i)=><li key={i} className="flex gap-2.5 text-sm text-slate-700"><span className="w-4 h-4 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center shrink-0 mt-0.5 text-xs">✓</span>{s}</li>)}</ul>
                      :<EmptyState icon="✦" message="No strengths data"/>}
                  </Card>
                  <Card>
                    <SectionHeading>Technical Skills</SectionHeading>
                    <div className="flex flex-wrap gap-2">
                      {(result.structured_resume_data?.technical_skills||[]).length>0?result.structured_resume_data.technical_skills.map((s,i)=><Tag key={i} label={s} variant="indigo"/>):<p className="text-sm text-slate-400">None detected</p>}
                    </div>
                  </Card>
                  <Card>
                    <SectionHeading>Tools & Platforms</SectionHeading>
                    <div className="flex flex-wrap gap-2">
                      {(result.structured_resume_data?.tools||[]).length>0?result.structured_resume_data.tools.map((t,i)=><Tag key={i} label={t} variant="blue"/>):<p className="text-sm text-slate-400">None detected</p>}
                    </div>
                  </Card>
                  <Card>
                    <SectionHeading>Soft Skills & Competencies</SectionHeading>
                    <div className="flex flex-wrap gap-2">
                      {(result.structured_resume_data?.soft_skills||[]).length>0?result.structured_resume_data.soft_skills.map((s,i)=><Tag key={i} label={s} variant="green"/>):<p className="text-sm text-slate-400">None detected</p>}
                    </div>
                  </Card>
                  <Card>
                    <SectionHeading>Areas to Improve</SectionHeading>
                    {(result.llm_analysis?.weaknesses||[]).length>0
                      ?<ul className="space-y-2.5">{result.llm_analysis.weaknesses.map((w,i)=><li key={i} className="flex gap-2.5 text-sm text-slate-700"><span className="w-4 h-4 rounded-full bg-amber-100 text-amber-600 flex items-center justify-center shrink-0 mt-0.5 text-xs">!</span>{w}</li>)}</ul>
                      :<EmptyState icon="✦" message="No weaknesses detected"/>}
                  </Card>
                </div>
                {(result.llm_analysis?.suggestions||[]).length>0&&(
                  <Card>
                    <SectionHeading>ATS Improvement Suggestions</SectionHeading>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {result.llm_analysis.suggestions.map((s,i)=><div key={i} className="flex gap-2.5 bg-slate-50 rounded-xl p-3.5 text-sm text-slate-700"><span className="text-indigo-400 shrink-0 mt-0.5">→</span>{s}</div>)}
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* ── MATCH TAB ── */}
            {activeTab==="match"&&(
              <div className="space-y-6">
                <Card>
                  <SectionHeading>Job Description</SectionHeading>
                  <textarea placeholder="Paste the full job description here..." value={jobDescription} onChange={e=>setJobDescription(e.target.value)} className="w-full border border-slate-200 rounded-xl p-3 text-sm h-36 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-400 text-slate-700 placeholder:text-slate-400"/>
                  <ErrorBanner message={jobMatchError}/>
                  <PrimaryBtn onClick={handleJobMatch} loading={jobMatchLoading} loadingLabel="Matching..." className="mt-4 bg-blue-600 hover:bg-blue-700 text-white">Analyze Job Match →</PrimaryBtn>
                </Card>

                {jobMatchResult&&(
                  <>
                    {/* Score cards */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="text-center">
                        <p className="text-xs text-slate-500 font-medium mb-2">Match Score</p>
                        <p className={`text-5xl font-bold ${matchPct>=70?"text-emerald-600":matchPct>=50?"text-amber-500":"text-red-500"}`}>{matchPct}%</p>
                        <div className="mt-2"><ScoreBadge score={matchPct}/></div>
                        <div className="mt-3 bg-slate-100 rounded-full h-2"><div className={`h-2 rounded-full transition-all ${matchPct>=70?"bg-emerald-500":matchPct>=50?"bg-amber-400":"bg-red-400"}`} style={{width:`${matchPct}%`}}/></div>
                      </Card>
                      <Card className="text-center">
                        <p className="text-xs text-slate-500 font-medium mb-2">Domain Alignment</p>
                        <p className={`text-3xl font-bold ${jobMatchResult.job_match_analysis?.domain_alignment==="Strong"?"text-emerald-600":jobMatchResult.job_match_analysis?.domain_alignment==="Moderate"?"text-amber-500":"text-red-500"}`}>{jobMatchResult.job_match_analysis?.domain_alignment}</p>
                        <p className="text-xs text-slate-400 mt-2">Similarity: {jobMatchResult.job_match_analysis?.domain_similarity_score}</p>
                      </Card>
                    </div>

                    {/* Recruiter Insights Panel */}
                    {jobMatchResult.match_insights&&(
                      <Card className="border-indigo-200 bg-indigo-50">
                        <SectionHeading>Recruiter Assessment</SectionHeading>
                        <p className="text-sm text-slate-700 mb-4">{jobMatchResult.match_insights.summary}</p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                          <div>
                            <p className="text-xs font-semibold text-emerald-700 uppercase tracking-wide mb-2">Strengths</p>
                            <ul className="space-y-1">{(jobMatchResult.match_insights.strengths||[]).map((s,i)=><li key={i} className="flex gap-2 text-sm text-slate-700"><span className="text-emerald-500">•</span>{s}</li>)}</ul>
                          </div>
                          <div>
                            <p className="text-xs font-semibold text-amber-700 uppercase tracking-wide mb-2">Potential Concerns</p>
                            <ul className="space-y-1">{(jobMatchResult.match_insights.concerns||[]).map((c,i)=><li key={i} className="flex gap-2 text-sm text-slate-700"><span className="text-amber-500">•</span>{c}</li>)}</ul>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-500 font-medium">Overall Recommendation:</span>
                          <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                            jobMatchResult.match_insights.recommendation==="Strong Candidate"?"bg-emerald-600 text-white":
                            jobMatchResult.match_insights.recommendation==="Good Candidate"?"bg-blue-600 text-white":
                            jobMatchResult.match_insights.recommendation==="Moderate Candidate"?"bg-amber-500 text-white":
                            "bg-red-500 text-white"
                          }`}>{jobMatchResult.match_insights.recommendation}</span>
                        </div>
                      </Card>
                    )}

                    {/* Tiered skills */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <Card>
                        <SectionHeading>✓ Matched Skills</SectionHeading>
                        <div className="flex flex-wrap gap-2">
                          {(jobMatchResult.job_match_analysis?.matched_skills||[]).length>0?jobMatchResult.job_match_analysis.matched_skills.map((s,i)=><Tag key={i} label={s} variant="green"/>):<EmptyState icon="○" message="No direct matches"/>}
                        </div>
                        {(jobMatchResult.job_match_analysis?.related_skills||[]).length>0&&(
                          <div className="mt-4">
                            <p className="text-xs font-semibold text-amber-600 uppercase tracking-wide mb-2">≈ Related Skills</p>
                            <div className="space-y-2">
                              {jobMatchResult.job_match_analysis.related_skills.map((r,i)=>(
                                <div key={i} className="bg-yellow-50 border border-yellow-200 rounded-lg px-3 py-2">
                                  <span className="text-xs font-medium text-yellow-800">≈ {r.skill}</span>
                                  {r.reason
                                    ? <p className="text-xs text-yellow-700 mt-1">{r.reason}</p>
                                    : r.related_to && r.related_to!=="inferred from experience" &&
                                      <p className="text-xs text-yellow-700 mt-1">Related to: {r.related_to}</p>
                                  }
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </Card>
                      <Card>
                        <SectionHeading>✗ Missing Skills</SectionHeading>
                        <div className="flex flex-wrap gap-2">
                          {(jobMatchResult.job_match_analysis?.missing_skills||[]).length>0?jobMatchResult.job_match_analysis.missing_skills.map((s,i)=><Tag key={i} label={s} variant="red"/>):<EmptyState icon="✓" message="No skill gaps detected"/>}
                        </div>
                      </Card>
                      <Card>
                        <SectionHeading>Matched Tools</SectionHeading>
                        <div className="flex flex-wrap gap-2">
                          {(jobMatchResult.job_match_analysis?.matched_tools||[]).length>0?jobMatchResult.job_match_analysis.matched_tools.map((t,i)=><Tag key={i} label={t} variant="blue"/>):<EmptyState icon="○" message="No tool matches"/>}
                        </div>
                        {(jobMatchResult.job_match_analysis?.related_tools||[]).length>0&&(
                          <div className="mt-4">
                            <p className="text-xs font-semibold text-amber-600 uppercase tracking-wide mb-2">≈ Related Tools</p>
                            <div className="flex flex-wrap gap-2">
                              {jobMatchResult.job_match_analysis.related_tools.map((r,i)=><span key={i} title={`Related to: ${r.related_to}`} className="inline-flex items-center gap-1 bg-yellow-50 text-yellow-800 border border-yellow-200 px-2.5 py-1 rounded-md text-xs font-medium cursor-help">≈ {r.skill}</span>)}
                            </div>
                          </div>
                        )}
                      </Card>
                      <Card>
                        <SectionHeading>Missing Tools</SectionHeading>
                        <div className="flex flex-wrap gap-2">
                          {(jobMatchResult.job_match_analysis?.missing_tools||[]).length>0?jobMatchResult.job_match_analysis.missing_tools.map((t,i)=><Tag key={i} label={t} variant="amber"/>):<EmptyState icon="✓" message="No tool gaps"/>}
                        </div>
                      </Card>
                    </div>

                    {(jobMatchResult.job_match_analysis?.recommendations||[]).length>0&&(
                      <Card>
                        <SectionHeading>Recommendations</SectionHeading>
                        <div className="space-y-2">
                          {jobMatchResult.job_match_analysis.recommendations.map((r,i)=><div key={i} className="flex gap-2.5 text-sm text-slate-700 py-2 border-b border-slate-100 last:border-0"><span className="text-blue-400 shrink-0">→</span>{r}</div>)}
                        </div>
                      </Card>
                    )}
                  </>
                )}
              </div>
            )}

            {/* ── REWRITE TAB ── */}
            {activeTab==="rewrite"&&(
              <div className="space-y-6">
                <Card>
                  <SectionHeading>Target Role</SectionHeading>
                  <input type="text" placeholder="e.g. Senior Fashion Designer, Sales Manager, Data Scientist" value={targetRole} onChange={e=>setTargetRole(e.target.value)} className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400 text-slate-700 placeholder:text-slate-400"/>
                  <ErrorBanner message={rewriteError}/>
                  <PrimaryBtn onClick={handleResumeRewrite} loading={rewriteLoading} loadingLabel="Rewriting for your target role..." className="mt-4 bg-purple-600 hover:bg-purple-700 text-white">Rewrite Resume →</PrimaryBtn>
                </Card>
                {rewriteResult&&(
                  <Card>
                    <SectionHeading>Optimized for: {rewriteResult.target_role}</SectionHeading>
                    <div className="space-y-2.5">
                      {(rewriteResult?.rewritten_resume?.optimized_bullets||[]).map((b,i)=>(
                        <div key={i} className="flex gap-3 bg-purple-50 rounded-xl p-3.5 text-sm text-slate-800"><span className="text-purple-400 shrink-0 mt-0.5 font-bold">•</span>{b}</div>
                      ))}
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* ── CHAT TAB ── */}
            {activeTab==="chat"&&(
              <div className="space-y-4">
                <Card className="p-0 overflow-hidden">
                  <div className="flex items-center gap-3 px-5 py-4 border-b border-slate-100">
                    <div className="w-8 h-8 rounded-xl bg-emerald-600 flex items-center justify-center"><svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg></div>
                    <div><p className="text-sm font-semibold text-slate-900">CareerPilot AI Mentor</p><p className="text-xs text-slate-400">Personalized career guidance based on your profile</p></div>
                  </div>
                  <div className="min-h-64 max-h-96 overflow-y-auto p-5 space-y-4 bg-slate-50">
                    {chatHistory.length===0
                      ?<div className="flex flex-col items-center justify-center h-40 text-slate-400"><div className="w-10 h-10 rounded-full bg-slate-200 flex items-center justify-center mb-3"><svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg></div><p className="text-sm">Ask anything about your career, skills, or resume</p></div>
                      :chatHistory.map((m,i)=>(
                        <div key={i} className={`flex ${m.role==="user"?"justify-end":"justify-start"}`}>
                          <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap ${m.role==="user"?"bg-indigo-600 text-white":m.role==="error"?"bg-red-50 text-red-700 border border-red-200":"bg-white text-slate-800 border border-slate-200"}`}>{m.text}</div>
                        </div>
                      ))
                    }
                    <div ref={chatEndRef}/>
                  </div>
                  <div className="px-5 py-4 border-t border-slate-100 bg-white">
                    <div className="flex gap-3">
                      <textarea placeholder="Ask about your career path, skill gaps, interview tips..." value={chatMessage} onChange={e=>setChatMessage(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();handleCareerChat();}}} rows={2} className="flex-1 border border-slate-200 rounded-xl px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-emerald-400 text-slate-700 placeholder:text-slate-400"/>
                      <button onClick={handleCareerChat} disabled={chatLoading||!chatMessage.trim()} className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2 self-end">
                        {chatLoading?<svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/></svg>:<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>}
                      </button>
                    </div>
                    <ErrorBanner message={chatError}/>
                  </div>
                </Card>
                {chatHistory.length===0&&(
                  <div className="grid grid-cols-2 gap-3">
                    {["What are my biggest resume weaknesses?","How can I improve my ATS score?","What skills should I learn next?","How do I transition to a new role?"].map((q,i)=>(
                      <button key={i} onClick={()=>setChatMessage(q)} className="text-left text-xs text-slate-600 bg-white border border-slate-200 rounded-xl p-3.5 hover:border-indigo-300 hover:bg-indigo-50 transition-all">{q}</button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* ── INTERVIEW TAB ── */}
            {activeTab==="interview"&&(
              <div className="space-y-6">
                <Card>
                  <SectionHeading>Generate Interview Questions</SectionHeading>
                  <input type="text" placeholder="e.g. Senior Fashion Designer, Data Scientist, Sales Manager" value={interviewRole} onChange={e=>setInterviewRole(e.target.value)} className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 text-slate-700 placeholder:text-slate-400"/>
                  <ErrorBanner message={interviewError}/>
                  <PrimaryBtn onClick={handleInterviewGenerator} loading={interviewLoading} loadingLabel="Generating personalized questions..." className="mt-4 bg-indigo-600 hover:bg-indigo-700 text-white">Generate Interview Questions →</PrimaryBtn>
                </Card>
                {interviewQuestions&&(
                  <div className="space-y-5">
                    {[
                      {key:"behavioral_questions",label:"Behavioral Questions",icon:"🤝",accent:"emerald"},
                      {key:"technical_questions",label:"Technical Questions",icon:"🔧",accent:"indigo"},
                      {key:"leadership_questions",label:"Leadership Questions",icon:"👑",accent:"purple"},
                      {key:"skill_gap_questions",label:"Skill Gap Questions",icon:"📈",accent:"amber"},
                      {key:"role_specific_questions",label:"Role-Specific Questions",icon:"🎯",accent:"blue"},
                    ].map(({key,label,icon,accent})=>{
                      const qs=interviewQuestions?.interview_questions?.[key]||[];
                      if(!qs.length)return null;
                      const bg={indigo:"bg-indigo-50 border-indigo-200 text-indigo-800",emerald:"bg-emerald-50 border-emerald-200 text-emerald-800",amber:"bg-amber-50 border-amber-200 text-amber-800",purple:"bg-purple-50 border-purple-200 text-purple-800",blue:"bg-blue-50 border-blue-200 text-blue-800"};
                      return(
                        <Card key={key}>
                          <SectionHeading>{icon} {label}</SectionHeading>
                          <div className="space-y-2.5">
                            {qs.map((q,i)=><div key={i} className={`rounded-xl border p-3.5 text-sm ${bg[accent]}`}><span className="font-semibold opacity-60 mr-2">Q{i+1}.</span>{q}</div>)}
                          </div>
                        </Card>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Reset */}
            <div className="mt-8 pt-6 border-t border-slate-200 flex items-center justify-between">
              <p className="text-xs text-slate-400">Analyzing: <span className="font-medium text-slate-600">{result.filename}</span></p>
              <button onClick={()=>{setResult(null);setFile(null);setJobMatchResult(null);setRewriteResult(null);setInterviewQuestions(null);setChatHistory([]);}} className="text-xs text-slate-500 hover:text-red-600 transition-colors font-medium">← Upload a different resume</button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
